# =======================================================================


defmodule ProteinAPI.GenServerTelemetry do
  @moduledoc """
  GenServer Telemetry events (queue depth, RTT, error rates)
  Comprehensive monitoring for all GenServer processes in the system
  """

  use GenServer
  require Logger

  defstruct [
    :monitored_processes,
    :metrics_history,
    :collection_interval,
    :max_history_size
  ]

  @collection_interval 1_000  # 1 second
  @max_history_size 1000
  @default_processes_to_monitor [
    ProteinAPI.ComputationManager,
    ProteinAPI.JuliaPort,
    ProteinAPI.MessagePackChannel,
    ProteinAPI.BackPressure,
    ProteinAPI.PriorityQueue,
    ProteinAPI.ETSOptimization
  ]

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def get_metrics(process_name \\ :all) do
    GenServer.call(__MODULE__, {:get_metrics, process_name})
  end

  def get_live_stats do
    GenServer.call(__MODULE__, :get_live_stats)
  end

  def add_monitored_process(process_name) do
    GenServer.cast(__MODULE__, {:add_monitored_process, process_name})
  end

  def remove_monitored_process(process_name) do
    GenServer.cast(__MODULE__, {:remove_monitored_process, process_name})
  end

  def init(opts) do
    collection_interval = Keyword.get(opts, :collection_interval, @collection_interval)
    processes_to_monitor = Keyword.get(opts, :processes, @default_processes_to_monitor)

    state = %__MODULE__{
      monitored_processes: MapSet.new(processes_to_monitor),
      metrics_history: %{},
      collection_interval: collection_interval,
      max_history_size: Keyword.get(opts, :max_history_size, @max_history_size)
    }

    # Start metric collection
    schedule_collection(collection_interval)

    # Attach telemetry handlers
    attach_telemetry_handlers()

    {:ok, state}
  end

  def handle_call({:get_metrics, process_name}, _from, state) do
    metrics = case process_name do
      :all -> state.metrics_history
      specific_process -> Map.get(state.metrics_history, specific_process, [])
    end

    {:reply, metrics, state}
  end

  def handle_call(:get_live_stats, _from, state) do
    live_stats = collect_live_stats(state.monitored_processes)
    {:reply, live_stats, state}
  end

  def handle_cast({:add_monitored_process, process_name}, state) do
    updated_processes = MapSet.put(state.monitored_processes, process_name)
    {:noreply, %{state | monitored_processes: updated_processes}}
  end

  def handle_cast({:remove_monitored_process, process_name}, state) do
    updated_processes = MapSet.delete(state.monitored_processes, process_name)
    updated_history = Map.delete(state.metrics_history, process_name)

    {:noreply, %{state |
      monitored_processes: updated_processes,
      metrics_history: updated_history
    }}
  end

  def handle_info(:collect_metrics, state) do
    # Collect metrics from all monitored processes
    timestamp = System.monotonic_time(:millisecond)

    updated_history = Enum.reduce(state.monitored_processes, state.metrics_history, fn process_name, acc ->
      case collect_process_metrics(process_name, timestamp) do
        {:ok, metrics} ->
          process_history = Map.get(acc, process_name, [])
          updated_process_history = [metrics | process_history]
            |> Enum.take(state.max_history_size)

          Map.put(acc, process_name, updated_process_history)

        {:error, reason} ->
          Logger.warn("Failed to collect metrics for #{process_name}: #{inspect(reason)}")
          acc
      end
    end)

    # Schedule next collection
    schedule_collection(state.collection_interval)

    {:noreply, %{state | metrics_history: updated_history}}
  end

  defp collect_process_metrics(process_name, timestamp) do
    case Process.whereis(process_name) do
      nil ->
        {:error, :process_not_found}

      pid ->
        try do
          # Get process info
          process_info = Process.info(pid, [
            :message_queue_len,
            :memory,
            :heap_size,
            :stack_size,
            :reductions,
            :status
          ])

          # Get custom metrics if available
          custom_metrics = get_custom_metrics(process_name)

          metrics = %{
            timestamp: timestamp,
            pid: pid,
            process_name: process_name,
            message_queue_len: process_info[:message_queue_len],
            memory_bytes: process_info[:memory],
            heap_size: process_info[:heap_size],
            stack_size: process_info[:stack_size],
            reductions: process_info[:reductions],
            status: process_info[:status],
            custom_metrics: custom_metrics
          }

          # Emit telemetry event
          :telemetry.execute(
            [:protein_api, :genserver, :metrics_collected],
            %{
              queue_depth: metrics.message_queue_len,
              memory_usage: metrics.memory_bytes,
              reductions: metrics.reductions
            },
            %{
              process_name: process_name,
              status: metrics.status
            }
          )

          {:ok, metrics}

        rescue
          error ->
            {:error, error}
        end
    end
  end

  defp get_custom_metrics(process_name) do
    try do
      case process_name do
        ProteinAPI.ComputationManager ->
          %{
            active_jobs: GenServer.call(process_name, :get_active_job_count, 1000),
            queue_depth: GenServer.call(process_name, :get_queue_depth, 1000),
            worker_pool_size: GenServer.call(process_name, :get_worker_count, 1000)
          }

        ProteinAPI.PriorityQueue ->
          case GenServer.call(process_name, :get_stats, 1000) do
            stats when is_map(stats) ->
              %{
                total_queue_size: stats[:current_queue_sizes][:total] || 0,
                priority_distribution: stats[:current_queue_sizes][:by_priority] || %{},
                average_wait_times: stats[:average_wait_times] || %{}
              }
            _ -> %{}
          end

        ProteinAPI.BackPressure ->
          case GenServer.call(process_name, :get_metrics, 1000) do
            metrics when is_map(metrics) ->
              %{
                current_load: metrics[:current_load] || 0,
                max_concurrent_jobs: metrics[:max_concurrent_jobs] || 0,
                utilization: metrics[:current_load] / max(metrics[:max_concurrent_jobs], 1),
                avg_response_time: metrics[:avg_response_time] || 0,
                error_rate: metrics[:error_rate] || 0
              }
            _ -> %{}
          end

        ProteinAPI.ETSOptimization ->
          case GenServer.call(process_name, :get_stats, 1000) do
            stats when is_map(stats) ->
              %{
                read_hit_rate: calculate_hit_rate(stats[:reads]),
                write_success_rate: calculate_success_rate(stats[:writes]),
                total_cache_entries: get_total_entries(stats[:table_sizes]),
                memory_usage_bytes: get_total_memory(stats[:memory_usage])
              }
            _ -> %{}
          end

        ProteinAPI.MessagePackChannel ->
          case GenServer.call(process_name, :get_encoding_stats, 1000) do
            stats when is_map(stats) ->
              %{
                compression_ratio: stats[:compression_ratio] || 0,
                encoding_time_ms: stats[:encoding_time_ms] || 0,
                msgpack_bytes: stats[:msgpack_bytes] || 0,
                json_bytes: stats[:json_bytes] || 0
              }
            _ -> %{}
          end

        _ ->
          %{}
      end
    rescue
      _ -> %{}
    catch
      :exit, _ -> %{}
    end
  end

  defp collect_live_stats(monitored_processes) do
    timestamp = System.monotonic_time(:millisecond)

    stats = Enum.map(monitored_processes, fn process_name ->
      case collect_process_metrics(process_name, timestamp) do
        {:ok, metrics} -> {process_name, metrics}
        {:error, _} -> {process_name, :unavailable}
      end
    end) |> Map.new()

    # Add system-wide stats
    system_stats = %{
      erlang_processes: :erlang.system_info(:process_count),
      erlang_ports: :erlang.system_info(:port_count),
      memory_total: :erlang.memory(:total),
      memory_processes: :erlang.memory(:processes),
      memory_ets: :erlang.memory(:ets),
      run_queue: :erlang.statistics(:run_queue)
    }

    Map.put(stats, :system, system_stats)
  end

  defp schedule_collection(interval) do
    Process.send_after(self(), :collect_metrics, interval)
  end

  defp attach_telemetry_handlers do
    events = [
      [:protein_api, :computation_manager, :job_started],
      [:protein_api, :computation_manager, :job_completed],
      [:protein_api, :julia_port, :command_sent],
      [:protein_api, :julia_port, :response_received],
      [:protein_api, :priority_queue, :enqueued],
      [:protein_api, :priority_queue, :dequeued],
      [:protein_api, :back_pressure, :limit_adjusted]
    ]

    :telemetry.attach_many(
      "genserver-telemetry-handler",
      events,
      &handle_telemetry_event/4,
      %{}
    )
  end

  defp handle_telemetry_event(event_name, measurements, metadata, _config) do
    # Calculate derived metrics
    case event_name do
      [:protein_api, :computation_manager, :job_completed] ->
        rtt = measurements[:duration] || 0

        :telemetry.execute(
          [:protein_api, :genserver, :rtt_recorded],
          %{rtt_microseconds: rtt},
          %{job_type: metadata[:job_type]}
        )

      [:protein_api, :julia_port, :response_received] ->
        rtt = measurements[:duration] || 0

        :telemetry.execute(
          [:protein_api, :genserver, :julia_rtt],
          %{rtt_microseconds: rtt},
          %{command_type: metadata[:command_type]}
        )

      [:protein_api, :priority_queue, :dequeued] ->
        wait_time = measurements[:wait_time] || 0

        :telemetry.execute(
          [:protein_api, :genserver, :queue_wait_time],
          %{wait_time_ms: wait_time},
          %{priority: metadata[:priority]}
        )

      _ ->
        :ok
    end
  end

  defp calculate_hit_rate(read_stats) when is_map(read_stats) do
    total = read_stats[:total] || 0
    hits = read_stats[:hits] || 0

    case total do
      0 -> 0.0
      _ -> hits / total
    end
  end
  defp calculate_hit_rate(_), do: 0.0

  defp calculate_success_rate(write_stats) when is_map(write_stats) do
    total = write_stats[:total] || 0
    success = write_stats[:success] || 0

    case total do
      0 -> 0.0
      _ -> success / total
    end
  end
  defp calculate_success_rate(_), do: 0.0

  defp get_total_entries(table_sizes) when is_map(table_sizes) do
    Enum.sum(for {_table, sizes} <- table_sizes, do: sizes[:total] || 0)
  end
  defp get_total_entries(_), do: 0

  defp get_total_memory(memory_usage) when is_map(memory_usage) do
    Enum.sum(for {_table, usage} <- memory_usage, do: usage[:total_bytes] || 0)
  end
  defp get_total_memory(_), do: 0
end

# =======================================================================


# =======================================================================
