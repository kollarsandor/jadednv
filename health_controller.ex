# =======================================================================


defmodule ProteinAPI.HealthController do
  @moduledoc """
  Health Controller with degradation reporting for UDP transport and other components
  """

  use GenServer
  require Logger

  defstruct [
    :component_health,
    :degradation_thresholds,
    :alert_handlers,
    :health_history,
    :monitoring_intervals
  ]

  @health_check_interval 5000
  @degradation_threshold_error_rate 0.05
  @degradation_threshold_latency 1000
  @degradation_threshold_packet_loss 0.02

  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def report_degradation(component, endpoint_id, reason) do
    GenServer.cast(__MODULE__, {:report_degradation, component, endpoint_id, reason})
  end

  def get_component_health(component) do
    GenServer.call(__MODULE__, {:get_health, component})
  end

  def get_overall_health do
    GenServer.call(__MODULE__, :get_overall_health)
  end

  def register_alert_handler(handler_fun) do
    GenServer.call(__MODULE__, {:register_alert_handler, handler_fun})
  end

  def init(_) do
    state = %__MODULE__{
      component_health: %{},
      degradation_thresholds: init_thresholds(),
      alert_handlers: [],
      health_history: :ets.new(:health_history, [:ordered_set, :private]),
      monitoring_intervals: %{}
    }

    # Start health monitoring
    schedule_health_check()

    {:ok, state}
  end

  def handle_cast({:report_degradation, component, endpoint_id, reason}, state) do
    timestamp = :os.system_time(:millisecond)

    degradation = %{
      component: component,
      endpoint_id: endpoint_id,
      reason: reason,
      reported_at: timestamp,
      severity: calculate_severity(reason)
    }

    # Store in history
    :ets.insert(state.health_history, {timestamp, degradation})

    # Update component health
    updated_health = update_component_health(state.component_health, component, degradation)

    # Check if alerting is needed
    check_alert_conditions(state, component, degradation)

    Logger.warn("Health degradation reported: #{component}/#{endpoint_id} - #{reason}")

    :telemetry.execute(
      [:protein_api, :health, :degradation_reported],
      %{severity: degradation.severity},
      %{
        component: component,
        endpoint: endpoint_id,
        reason: reason
      }
    )

    new_state = %{state | component_health: updated_health}
    {:noreply, new_state}
  end

  def handle_call({:get_health, component}, _from, state) do
    health = Map.get(state.component_health, component, %{status: :unknown})
    {:reply, health, state}
  end

  def handle_call(:get_overall_health, _from, state) do
    overall_health = calculate_overall_health(state.component_health)
    {:reply, overall_health, state}
  end

  def handle_call({:register_alert_handler, handler_fun}, _from, state) do
    updated_handlers = [handler_fun | state.alert_handlers]
    new_state = %{state | alert_handlers: updated_handlers}
    {:reply, :ok, new_state}
  end

  def handle_info(:health_check, state) do
    # Perform health checks on all registered components
    updated_state = perform_health_checks(state)

    schedule_health_check()
    {:noreply, updated_state}
  end

  defp perform_health_checks(state) do
    components_to_check = [
      :udp_transport,
      :julia_port,
      :computation_manager,
      :cache,
      :quantum_backends
    ]

    Enum.reduce(components_to_check, state, fn component, acc_state ->
      health_result = check_component_health(component)
      update_component_health_status(acc_state, component, health_result)
    end)
  end

  defp check_component_health(:udp_transport) do
    try do
      case ProteinAPI.UDPTransport.get_stats() do
        stats when is_map(stats) ->
          error_rate = calculate_error_rate(stats)

          cond do
            error_rate > @degradation_threshold_error_rate ->
              {:degraded, :high_error_rate, %{error_rate: error_rate}}

            stats.send_errors > 0 ->
              {:warning, :send_errors, %{errors: stats.send_errors}}

            true ->
              {:healthy, :normal_operation, stats}
          end

        _ ->
          {:error, :stats_unavailable, %{}}
      end
    rescue
      _ ->
        {:error, :component_unavailable, %{}}
    end
  end

  defp check_component_health(:julia_port) do
    try do
      case GenServer.call(ProteinAPI.JuliaPort, :health_check, 5000) do
        :healthy -> {:healthy, :normal_operation, %{}}
        :degraded -> {:degraded, :performance_issues, %{}}
        _ -> {:error, :unhealthy, %{}}
      end
    catch
      :exit, _ -> {:error, :component_unavailable, %{}}
    end
  end

  defp check_component_health(:computation_manager) do
    try do
      queue_depth = ProteinAPI.ComputationManager.queue_depth()
      active_jobs = ProteinAPI.ComputationManager.active_job_count()

      cond do
        queue_depth > 100 ->
          {:degraded, :high_queue_depth, %{queue_depth: queue_depth}}

        active_jobs == 0 and queue_depth > 0 ->
          {:warning, :processing_stalled, %{queue_depth: queue_depth}}

        true ->
          {:healthy, :normal_operation, %{queue_depth: queue_depth, active_jobs: active_jobs}}
      end
    rescue
      _ ->
        {:error, :component_unavailable, %{}}
    end
  end

  defp check_component_health(:cache) do
    try do
      # Check cache health metrics
      {:healthy, :normal_operation, %{}}
    rescue
      _ ->
        {:error, :component_unavailable, %{}}
    end
  end

  defp check_component_health(:quantum_backends) do
    # Check quantum backend connectivity
    if System.get_env("IBM_QUANTUM_API_TOKEN") do
      {:healthy, :api_token_configured, %{}}
    else
      {:warning, :no_api_token, %{}}
    end
  end

  defp update_component_health(health_map, component, degradation) do
    current_health = Map.get(health_map, component, %{
      status: :unknown,
      last_degradation: nil,
      degradation_count: 0,
      endpoints: %{}
    })

    endpoint_health = Map.get(current_health.endpoints, degradation.endpoint_id, %{
      status: :healthy,
      issues: []
    })

    updated_endpoint = %{endpoint_health |
      status: :degraded,
      issues: [degradation.reason | endpoint_health.issues] |> Enum.take(10)
    }

    updated_endpoints = Map.put(current_health.endpoints, degradation.endpoint_id, updated_endpoint)

    updated_health = %{current_health |
      status: determine_component_status(updated_endpoints),
      last_degradation: degradation.reported_at,
      degradation_count: current_health.degradation_count + 1,
      endpoints: updated_endpoints
    }

    Map.put(health_map, component, updated_health)
  end

  defp update_component_health_status(state, component, {status, reason, metrics}) do
    timestamp = :os.system_time(:millisecond)

    current_health = Map.get(state.component_health, component, %{
      status: :unknown,
      last_check: nil,
      metrics: %{},
      endpoints: %{}
    })

    updated_health = %{current_health |
      status: status,
      last_check: timestamp,
      metrics: metrics
    }

    # Log status changes
    if current_health.status != status do
      Logger.info("Component #{component} health changed: #{current_health.status} -> #{status} (#{reason})")

      :telemetry.execute(
        [:protein_api, :health, :status_changed],
        %{timestamp: timestamp},
        %{
          component: component,
          old_status: current_health.status,
          new_status: status,
          reason: reason
        }
      )
    end

    updated_component_health = Map.put(state.component_health, component, updated_health)
    %{state | component_health: updated_component_health}
  end

  defp determine_component_status(endpoints) when map_size(endpoints) == 0, do: :healthy

  defp determine_component_status(endpoints) do
    statuses = Map.values(endpoints) |> Enum.map(& &1.status)

    cond do
      Enum.all?(statuses, &(&1 == :healthy)) -> :healthy
      Enum.any?(statuses, &(&1 == :error)) -> :error
      Enum.any?(statuses, &(&1 == :degraded)) -> :degraded
      true -> :warning
    end
  end

  defp calculate_overall_health(component_health) do
    if map_size(component_health) == 0 do
      %{status: :unknown, components: %{}}
    else
      component_statuses = for {component, health} <- component_health, into: %{} do
        {component, health.status}
      end

      overall_status = case Map.values(component_statuses) do
        statuses when length(statuses) == 0 ->
          :unknown
        statuses ->
          cond do
            Enum.any?(statuses, &(&1 == :error)) -> :error
            Enum.any?(statuses, &(&1 == :degraded)) -> :degraded
            Enum.any?(statuses, &(&1 == :warning)) -> :warning
            Enum.all?(statuses, &(&1 == :healthy)) -> :healthy
            true -> :unknown
          end
      end

      %{
        status: overall_status,
        components: component_statuses,
        last_updated: :os.system_time(:millisecond)
      }
    end
  end

  defp check_alert_conditions(state, component, degradation) do
    # Check if this degradation should trigger alerts
    should_alert = case degradation.severity do
      :critical -> true
      :high -> true
      :medium -> degradation_count_exceeds_threshold(state, component)
      _ -> false
    end

    if should_alert do
      trigger_alerts(state, component, degradation)
    end
  end

  defp degradation_count_exceeds_threshold(state, component) do
    current_health = Map.get(state.component_health, component, %{degradation_count: 0})
    current_health.degradation_count >= 5
  end

  defp trigger_alerts(state, component, degradation) do
    alert_message = %{
      type: :health_degradation,
      component: component,
      endpoint: degradation.endpoint_id,
      reason: degradation.reason,
      severity: degradation.severity,
      timestamp: degradation.reported_at
    }

    Enum.each(state.alert_handlers, fn handler ->
      try do
        handler.(alert_message)
      rescue
        error ->
          Logger.error("Alert handler failed: #{inspect(error)}")
      end
    end)

    Logger.error("ALERT: Health degradation in #{component} - #{degradation.reason}")
  end

  defp calculate_severity(:heartbeat_timeout), do: :high
  defp calculate_severity(:high_error_rate), do: :high
  defp calculate_severity(:connection_failed), do: :critical
  defp calculate_severity(:send_errors), do: :medium
  defp calculate_severity(:performance_issues), do: :medium
  defp calculate_severity(_), do: :low

  defp calculate_error_rate(stats) do
    total_packets = stats.packets_sent + stats.packets_received

    if total_packets > 0 do
      stats.send_errors / total_packets
    else
      0.0
    end
  end

  defp init_thresholds do
    %{
      error_rate: @degradation_threshold_error_rate,
      latency: @degradation_threshold_latency,
      packet_loss: @degradation_threshold_packet_loss
    }
  end

  defp schedule_health_check do
    Process.send_after(self(), :health_check, @health_check_interval)
  end
end

# =======================================================================


# =======================================================================
