# =======================================================================


defmodule ProteinAPI.BackPressure do
  @moduledoc """
  Back-pressure module with dynamic max_concurrent_jobs adjustment
  Prevents system overload and maintains optimal throughput
  """

  use GenServer
  require Logger

  defstruct [
    :current_load,
    :max_concurrent_jobs,
    :base_max_jobs,
    :queue_depth,
    :response_times,
    :error_rates,
    :adjustment_history,
    :last_adjustment,
    :metrics_window
  ]

  @adjustment_interval 10_000  # 10 seconds
  @metrics_window_size 100
  @min_jobs 1
  @max_jobs 500
  @load_threshold_high 0.8
  @load_threshold_low 0.4
  @error_rate_threshold 0.05
  @response_time_threshold 5000  # 5 seconds

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def check_capacity do
    GenServer.call(__MODULE__, :check_capacity)
  end

  def record_job_start do
    GenServer.cast(__MODULE__, :job_started)
  end

  def record_job_completion(response_time, success) do
    GenServer.cast(__MODULE__, {:job_completed, response_time, success})
  end

  def get_current_limits do
    GenServer.call(__MODULE__, :get_limits)
  end

  def get_metrics do
    GenServer.call(__MODULE__, :get_metrics)
  end

  def init(opts) do
    base_max_jobs = Keyword.get(opts, :max_concurrent_jobs, 50)

    state = %__MODULE__{
      current_load: 0,
      max_concurrent_jobs: base_max_jobs,
      base_max_jobs: base_max_jobs,
      queue_depth: 0,
      response_times: :queue.new(),
      error_rates: :queue.new(),
      adjustment_history: [],
      last_adjustment: System.monotonic_time(:millisecond),
      metrics_window: %{
        response_times: [],
        error_count: 0,
        success_count: 0,
        queue_depths: []
      }
    }

    # Schedule periodic adjustments
    Process.send_after(self(), :adjust_limits, @adjustment_interval)

    {:ok, state}
  end

  def handle_call(:check_capacity, _from, state) do
    capacity_available = state.current_load < state.max_concurrent_jobs

    response = if capacity_available do
      {:ok, state.max_concurrent_jobs - state.current_load}
    else
      queue_manager = ProteinAPI.ComputationManager
      queue_depth = GenServer.call(queue_manager, :get_queue_depth)

      estimated_wait = calculate_estimated_wait(state, queue_depth)
      {:backpressure, %{
        queue_depth: queue_depth,
        estimated_wait_ms: estimated_wait,
        current_load: state.current_load,
        max_jobs: state.max_concurrent_jobs
      }}
    end

    {:reply, response, state}
  end

  def handle_call(:get_limits, _from, state) do
    limits = %{
      max_concurrent_jobs: state.max_concurrent_jobs,
      current_load: state.current_load,
      utilization: state.current_load / state.max_concurrent_jobs
    }
    {:reply, limits, state}
  end

  def handle_call(:get_metrics, _from, state) do
    metrics = %{
      current_load: state.current_load,
      max_concurrent_jobs: state.max_concurrent_jobs,
      queue_depth: state.queue_depth,
      avg_response_time: calculate_avg_response_time(state),
      error_rate: calculate_error_rate(state),
      adjustment_history: Enum.take(state.adjustment_history, 10)
    }
    {:reply, metrics, state}
  end

  def handle_cast(:job_started, state) do
    updated_state = %{state | current_load: state.current_load + 1}

    :telemetry.execute(
      [:protein_api, :back_pressure, :job_started],
      %{current_load: updated_state.current_load},
      %{max_jobs: state.max_concurrent_jobs}
    )

    {:noreply, updated_state}
  end

  def handle_cast({:job_completed, response_time, success}, state) do
    # Update current load
    updated_load = max(0, state.current_load - 1)

    # Update metrics window
    updated_metrics = update_metrics_window(state.metrics_window, response_time, success)

    updated_state = %{state |
      current_load: updated_load,
      metrics_window: updated_metrics
    }

    :telemetry.execute(
      [:protein_api, :back_pressure, :job_completed],
      %{
        current_load: updated_load,
        response_time: response_time
      },
      %{
        success: success,
        max_jobs: state.max_concurrent_jobs
      }
    )

    {:noreply, updated_state}
  end

  def handle_info(:adjust_limits, state) do
    {new_max_jobs, adjustment_reason} = calculate_new_limits(state)

    updated_state = if new_max_jobs != state.max_concurrent_jobs do
      Logger.info("Adjusting max_concurrent_jobs: #{state.max_concurrent_jobs} -> #{new_max_jobs} (#{adjustment_reason})")

      adjustment = %{
        timestamp: System.monotonic_time(:millisecond),
        old_limit: state.max_concurrent_jobs,
        new_limit: new_max_jobs,
        reason: adjustment_reason,
        metrics: %{
          avg_response_time: calculate_avg_response_time(state),
          error_rate: calculate_error_rate(state),
          utilization: state.current_load / state.max_concurrent_jobs
        }
      }

      :telemetry.execute(
        [:protein_api, :back_pressure, :limit_adjusted],
        %{
          old_limit: state.max_concurrent_jobs,
          new_limit: new_max_jobs
        },
        %{reason: adjustment_reason}
      )

      %{state |
        max_concurrent_jobs: new_max_jobs,
        adjustment_history: [adjustment | state.adjustment_history],
        last_adjustment: System.monotonic_time(:millisecond)
      }
    else
      state
    end

    # Schedule next adjustment
    Process.send_after(self(), :adjust_limits, @adjustment_interval)

    {:noreply, updated_state}
  end

  defp calculate_new_limits(state) do
    avg_response_time = calculate_avg_response_time(state)
    error_rate = calculate_error_rate(state)
    utilization = state.current_load / state.max_concurrent_jobs

    cond do
      # Decrease if error rate is too high
      error_rate > @error_rate_threshold ->
        new_limit = max(@min_jobs, round(state.max_concurrent_jobs * 0.8))
        {new_limit, "high_error_rate"}

      # Decrease if response time is too high
      avg_response_time > @response_time_threshold ->
        new_limit = max(@min_jobs, round(state.max_concurrent_jobs * 0.9))
        {new_limit, "high_response_time"}

      # Increase if utilization is high but performance is good
      utilization > @load_threshold_high and error_rate < @error_rate_threshold / 2 and
      avg_response_time < @response_time_threshold / 2 ->
        new_limit = min(@max_jobs, round(state.max_concurrent_jobs * 1.2))
        {new_limit, "good_performance_high_utilization"}

      # Moderate increase if utilization is medium and performance is excellent
      utilization > @load_threshold_low and utilization < @load_threshold_high and
      error_rate < @error_rate_threshold / 4 and avg_response_time < @response_time_threshold / 4 ->
        new_limit = min(@max_jobs, round(state.max_concurrent_jobs * 1.1))
        {new_limit, "excellent_performance"}

      # No change needed
      true ->
        {state.max_concurrent_jobs, "no_change"}
    end
  end

  defp calculate_avg_response_time(state) do
    case state.metrics_window.response_times do
      [] -> 0
      times -> Enum.sum(times) / length(times)
    end
  end

  defp calculate_error_rate(state) do
    total = state.metrics_window.error_count + state.metrics_window.success_count
    case total do
      0 -> 0.0
      _ -> state.metrics_window.error_count / total
    end
  end

  defp update_metrics_window(metrics, response_time, success) do
    updated_times = [response_time | metrics.response_times]
      |> Enum.take(@metrics_window_size)

    {error_count, success_count} = if success do
      {metrics.error_count, metrics.success_count + 1}
    else
      {metrics.error_count + 1, metrics.success_count}
    end

    %{metrics |
      response_times: updated_times,
      error_count: min(error_count, @metrics_window_size),
      success_count: min(success_count, @metrics_window_size)
    }
  end

  defp calculate_estimated_wait(state, queue_depth) do
    avg_response_time = calculate_avg_response_time(state)
    processing_rate = state.max_concurrent_jobs / max(avg_response_time, 1)

    round(queue_depth / processing_rate)
  end
end

# =======================================================================


# =======================================================================
