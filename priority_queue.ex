# =======================================================================


defmodule ProteinAPI.PriorityQueue do
  @moduledoc """
  Priority queue (fast tasks get precedence)
  Multi-level queue system with dynamic priority adjustment
  """

  use GenServer
  require Logger

  defstruct [
    :queues,
    :priority_levels,
    :stats,
    :preemption_enabled,
    :dynamic_priority_enabled
  ]

  @priority_levels [:urgent, :high, :normal, :low, :background]
  @max_queue_size 10_000
  @preemption_threshold 100  # milliseconds
  @dynamic_adjustment_interval 5_000  # 5 seconds

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def enqueue(item, priority \\ :normal, opts \\ []) do
    estimated_duration = Keyword.get(opts, :estimated_duration, 1000)
    job_type = Keyword.get(opts, :job_type, :default)
    deadline = Keyword.get(opts, :deadline)

    GenServer.call(__MODULE__, {:enqueue, item, priority, estimated_duration, job_type, deadline})
  end

  def dequeue(timeout \\ 5000) do
    GenServer.call(__MODULE__, :dequeue, timeout)
  end

  def peek do
    GenServer.call(__MODULE__, :peek)
  end

  def size do
    GenServer.call(__MODULE__, :size)
  end

  def get_stats do
    GenServer.call(__MODULE__, :get_stats)
  end

  def update_job_progress(job_id, progress) do
    GenServer.cast(__MODULE__, {:update_progress, job_id, progress})
  end

  def init(opts) do
    preemption_enabled = Keyword.get(opts, :preemption_enabled, true)
    dynamic_priority_enabled = Keyword.get(opts, :dynamic_priority_enabled, true)

    # Initialize priority queues
    queues = for level <- @priority_levels, into: %{} do
      {level, :queue.new()}
    end

    state = %__MODULE__{
      queues: queues,
      priority_levels: @priority_levels,
      stats: initialize_stats(),
      preemption_enabled: preemption_enabled,
      dynamic_priority_enabled: dynamic_priority_enabled
    }

    # Schedule dynamic priority adjustments
    if dynamic_priority_enabled do
      Process.send_after(self(), :adjust_priorities, @dynamic_adjustment_interval)
    end

    {:ok, state}
  end

  def handle_call({:enqueue, item, initial_priority, estimated_duration, job_type, deadline}, _from, state) do
    # Calculate effective priority based on multiple factors
    effective_priority = calculate_effective_priority(
      initial_priority,
      estimated_duration,
      job_type,
      deadline,
      state
    )

    job_id = generate_job_id()

    job_item = %{
      id: job_id,
      data: item,
      initial_priority: initial_priority,
      effective_priority: effective_priority,
      estimated_duration: estimated_duration,
      job_type: job_type,
      deadline: deadline,
      enqueued_at: System.monotonic_time(:millisecond),
      priority_adjustments: []
    }

    # Check queue capacity
    total_size = get_total_queue_size(state)
    if total_size >= @max_queue_size do
      {:reply, {:error, :queue_full}, state}
    else
      # Add to appropriate priority queue
      updated_queues = Map.update!(state.queues, effective_priority, fn queue ->
        :queue.in(job_item, queue)
      end)

      # Update statistics
      updated_stats = update_enqueue_stats(state.stats, effective_priority, estimated_duration)

      :telemetry.execute(
        [:protein_api, :priority_queue, :enqueued],
        %{
          queue_size: total_size + 1,
          estimated_duration: estimated_duration
        },
        %{
          initial_priority: initial_priority,
          effective_priority: effective_priority,
          job_type: job_type,
          job_id: job_id
        }
      )

      updated_state = %{state |
        queues: updated_queues,
        stats: updated_stats
      }

      {:reply, {:ok, job_id}, updated_state}
    end
  end

  def handle_call(:dequeue, _from, state) do
    case find_next_job(state) do
      {:ok, job_item, updated_queues} ->
        updated_stats = update_dequeue_stats(state.stats, job_item.effective_priority)

        :telemetry.execute(
          [:protein_api, :priority_queue, :dequeued],
          %{
            wait_time: System.monotonic_time(:millisecond) - job_item.enqueued_at,
            estimated_duration: job_item.estimated_duration
          },
          %{
            priority: job_item.effective_priority,
            job_type: job_item.job_type,
            job_id: job_item.id
          }
        )

        updated_state = %{state |
          queues: updated_queues,
          stats: updated_stats
        }

        {:reply, {:ok, job_item}, updated_state}

      {:empty} ->
        {:reply, {:error, :empty}, state}
    end
  end

  def handle_call(:peek, _from, state) do
    case find_next_job_without_removing(state) do
      {:ok, job_item} ->
        {:reply, {:ok, job_item}, state}
      {:empty} ->
        {:reply, {:error, :empty}, state}
    end
  end

  def handle_call(:size, _from, state) do
    total_size = get_total_queue_size(state)
    size_by_priority = for {priority, queue} <- state.queues, into: %{} do
      {priority, :queue.len(queue)}
    end

    response = %{
      total: total_size,
      by_priority: size_by_priority
    }

    {:reply, response, state}
  end

  def handle_call(:get_stats, _from, state) do
    enriched_stats = Map.merge(state.stats, %{
      current_queue_sizes: get_queue_sizes(state),
      average_wait_times: calculate_average_wait_times(state)
    })

    {:reply, enriched_stats, state}
  end

  def handle_cast({:update_progress, job_id, progress}, state) do
    # This could be used for preemption decisions in the future
    :telemetry.execute(
      [:protein_api, :priority_queue, :progress_updated],
      %{progress: progress},
      %{job_id: job_id}
    )

    {:noreply, state}
  end

  def handle_info(:adjust_priorities, state) do
    if state.dynamic_priority_enabled do
      updated_state = perform_dynamic_priority_adjustment(state)

      # Schedule next adjustment
      Process.send_after(self(), :adjust_priorities, @dynamic_adjustment_interval)

      {:noreply, updated_state}
    else
      {:noreply, state}
    end
  end

  defp calculate_effective_priority(initial_priority, estimated_duration, job_type, deadline, state) do
    # Fast tasks get priority boost
    duration_factor = case estimated_duration do
      d when d < 1000 -> 2    # Very fast tasks
      d when d < 5000 -> 1    # Fast tasks
      d when d < 30000 -> 0   # Normal tasks
      _ -> -1                 # Slow tasks
    end

    # Job type considerations
    type_factor = case job_type do
      :interactive -> 2
      :api_request -> 1
      :batch -> -1
      :background -> -2
      _ -> 0
    end

    # Deadline pressure
    deadline_factor = case deadline do
      nil -> 0
      deadline_ms ->
        time_remaining = deadline_ms - System.system_time(:millisecond)
        cond do
          time_remaining < 10_000 -> 3    # Very urgent
          time_remaining < 60_000 -> 2    # Urgent
          time_remaining < 300_000 -> 1   # Moderately urgent
          _ -> 0
        end
    end

    # Queue pressure factor
    queue_pressure = calculate_queue_pressure(state)
    pressure_factor = case queue_pressure do
      p when p > 0.8 -> 1   # High pressure, boost interactive tasks
      p when p > 0.5 -> 0   # Medium pressure
      _ -> 0                # Low pressure
    end

    # Calculate adjustment
    total_adjustment = duration_factor + type_factor + deadline_factor + pressure_factor

    # Apply adjustment to initial priority
    adjusted_priority = adjust_priority_level(initial_priority, total_adjustment)

    Logger.debug("Priority calculation: #{initial_priority} -> #{adjusted_priority} (factors: duration=#{duration_factor}, type=#{type_factor}, deadline=#{deadline_factor}, pressure=#{pressure_factor})")

    adjusted_priority
  end

  defp adjust_priority_level(current_priority, adjustment) do
    current_index = Enum.find_index(@priority_levels, &(&1 == current_priority)) || 2
    new_index = max(0, min(length(@priority_levels) - 1, current_index - adjustment))
    Enum.at(@priority_levels, new_index)
  end

  defp find_next_job(state) do
    # Check each priority level in order
    Enum.reduce_while(@priority_levels, {:empty}, fn priority, acc ->
      case :queue.out(state.queues[priority]) do
        {{:value, job_item}, updated_queue} ->
          updated_queues = Map.put(state.queues, priority, updated_queue)
          {:halt, {:ok, job_item, updated_queues}}
        {:empty, _} ->
          {:cont, acc}
      end
    end)
  end

  defp find_next_job_without_removing(state) do
    Enum.reduce_while(@priority_levels, {:empty}, fn priority, acc ->
      case :queue.peek(state.queues[priority]) do
        {:value, job_item} ->
          {:halt, {:ok, job_item}}
        :empty ->
          {:cont, acc}
      end
    end)
  end

  defp get_total_queue_size(state) do
    Enum.sum(for {_priority, queue} <- state.queues, do: :queue.len(queue))
  end

  defp calculate_queue_pressure(state) do
    total_size = get_total_queue_size(state)
    total_size / @max_queue_size
  end

  defp generate_job_id do
    :crypto.strong_rand_bytes(16) |> Base.url_encode64(padding: false)
  end

  defp initialize_stats do
    %{
      enqueued: %{total: 0, by_priority: Map.new(@priority_levels, &{&1, 0})},
      dequeued: %{total: 0, by_priority: Map.new(@priority_levels, &{&1, 0})},
      total_wait_time: 0,
      priority_adjustments: 0
    }
  end

  defp update_enqueue_stats(stats, priority, _estimated_duration) do
    %{stats |
      enqueued: %{
        total: stats.enqueued.total + 1,
        by_priority: Map.update!(stats.enqueued.by_priority, priority, &(&1 + 1))
      }
    }
  end

  defp update_dequeue_stats(stats, priority) do
    %{stats |
      dequeued: %{
        total: stats.dequeued.total + 1,
        by_priority: Map.update!(stats.dequeued.by_priority, priority, &(&1 + 1))
      }
    }
  end

  defp get_queue_sizes(state) do
    for {priority, queue} <- state.queues, into: %{} do
      {priority, :queue.len(queue)}
    end
  end

  defp calculate_average_wait_times(state) do
    current_time = System.monotonic_time(:millisecond)

    for {priority, queue} <- state.queues, into: %{} do
      items = :queue.to_list(queue)

      avg_wait = case items do
        [] -> 0
        items ->
          total_wait = Enum.sum(for item <- items, do: current_time - item.enqueued_at)
          total_wait / length(items)
      end

      {priority, avg_wait}
    end
  end

  defp perform_dynamic_priority_adjustment(state) do
    # This is a placeholder for more sophisticated dynamic priority adjustment
    # Could include analysis of queue depths, completion rates, etc.

    :telemetry.execute(
      [:protein_api, :priority_queue, :priority_adjustment],
      %{adjustments_made: 0},
      %{}
    )

    state
  end
end

# =======================================================================


# =======================================================================
