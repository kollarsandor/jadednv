# =======================================================================


defmodule ProteinAPI.ETSOptimization do
  @moduledoc """
  ETS read/write concurrency and table splitting (hot read vs write)
  Optimized for high-throughput concurrent access patterns
  """

  use GenServer
  require Logger

  defstruct [
    :read_tables,
    :write_tables,
    :table_config,
    :shard_count,
    :stats
  ]

  @default_shard_count 16
  @read_concurrency true
  @write_concurrency true
  @compressed true

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def put(key, value, opts \\ []) do
    table_type = Keyword.get(opts, :table_type, :default)
    ttl = Keyword.get(opts, :ttl)

    GenServer.call(__MODULE__, {:put, table_type, key, value, ttl})
  end

  def get(key, opts \\ []) do
    table_type = Keyword.get(opts, :table_type, :default)
    GenServer.call(__MODULE__, {:get, table_type, key})
  end

  def delete(key, opts \\ []) do
    table_type = Keyword.get(opts, :table_type, :default)
    GenServer.call(__MODULE__, {:delete, table_type, key})
  end

  def get_stats do
    GenServer.call(__MODULE__, :get_stats)
  end

  def get_table_info do
    GenServer.call(__MODULE__, :get_table_info)
  end

  def init(opts) do
    shard_count = Keyword.get(opts, :shard_count, @default_shard_count)

    # Define table configurations for different use cases
    table_configs = %{
      # Hot read cache (protein structures, sequences)
      hot_read: %{
        read_concurrency: true,
        write_concurrency: false,
        compressed: true,
        shard_count: shard_count * 2  # More shards for hot reads
      },

      # Write-heavy cache (job queue, temporary results)
      write_heavy: %{
        read_concurrency: false,
        write_concurrency: true,
        compressed: false,
        shard_count: shard_count
      },

      # Balanced cache (general purpose)
      default: %{
        read_concurrency: true,
        write_concurrency: true,
        compressed: true,
        shard_count: shard_count
      },

      # Large objects cache (folded structures, complex results)
      large_objects: %{
        read_concurrency: true,
        write_concurrency: false,
        compressed: true,
        shard_count: shard_count / 2  # Fewer shards for large objects
      }
    }

    # Create tables for each configuration
    {read_tables, write_tables} = create_table_shards(table_configs)

    state = %__MODULE__{
      read_tables: read_tables,
      write_tables: write_tables,
      table_config: table_configs,
      shard_count: shard_count,
      stats: initialize_stats()
    }

    {:ok, state}
  end

  def handle_call({:put, table_type, key, value, ttl}, _from, state) do
    start_time = System.monotonic_time(:microsecond)

    # Select appropriate table shard
    shard_index = shard_for_key(key, table_type, state)
    table = get_write_table(table_type, shard_index, state)

    # Prepare the record
    record = case ttl do
      nil -> {key, value, :no_ttl}
      ttl_ms -> {key, value, System.system_time(:millisecond) + ttl_ms}
    end

    # Insert with error handling
    result = try do
      :ets.insert(table, record)
      :ok
    rescue
      error ->
        Logger.error("ETS insert failed: #{inspect(error)}")
        {:error, :insert_failed}
    end

    # Update statistics
    duration = System.monotonic_time(:microsecond) - start_time
    updated_stats = update_stats(state.stats, :write, table_type, duration, result == :ok)

    :telemetry.execute(
      [:protein_api, :ets, :write],
      %{duration: duration, size: byte_size(:erlang.term_to_binary(value))},
      %{table_type: table_type, shard: shard_index, success: result == :ok}
    )

    {:reply, result, %{state | stats: updated_stats}}
  end

  def handle_call({:get, table_type, key}, _from, state) do
    start_time = System.monotonic_time(:microsecond)

    # Try read-optimized tables first
    shard_index = shard_for_key(key, table_type, state)
    read_table = get_read_table(table_type, shard_index, state)

    result = case :ets.lookup(read_table, key) do
      [{^key, value, :no_ttl}] ->
        {:ok, value}

      [{^key, value, expires_at}] ->
        current_time = System.system_time(:millisecond)
        if current_time < expires_at do
          {:ok, value}
        else
          # Clean up expired entry
          :ets.delete(read_table, key)
          {:error, :not_found}
        end

      [] ->
        # Fall back to write table if not found in read table
        write_table = get_write_table(table_type, shard_index, state)
        case :ets.lookup(write_table, key) do
          [{^key, value, :no_ttl}] ->
            # Promote to read table for future fast access
            :ets.insert(read_table, {key, value, :no_ttl})
            {:ok, value}

          [{^key, value, expires_at}] ->
            current_time = System.system_time(:millisecond)
            if current_time < expires_at do
              :ets.insert(read_table, {key, value, expires_at})
              {:ok, value}
            else
              :ets.delete(write_table, key)
              {:error, :not_found}
            end

          [] ->
            {:error, :not_found}
        end
    end

    # Update statistics
    duration = System.monotonic_time(:microsecond) - start_time
    updated_stats = update_stats(state.stats, :read, table_type, duration, match?({:ok, _}, result))

    :telemetry.execute(
      [:protein_api, :ets, :read],
      %{duration: duration},
      %{table_type: table_type, shard: shard_index, hit: match?({:ok, _}, result)}
    )

    {:reply, result, %{state | stats: updated_stats}}
  end

  def handle_call({:delete, table_type, key}, _from, state) do
    start_time = System.monotonic_time(:microsecond)

    shard_index = shard_for_key(key, table_type, state)
    read_table = get_read_table(table_type, shard_index, state)
    write_table = get_write_table(table_type, shard_index, state)

    # Delete from both tables
    read_deleted = :ets.delete(read_table, key)
    write_deleted = :ets.delete(write_table, key)

    result = if read_deleted or write_deleted, do: :ok, else: {:error, :not_found}

    duration = System.monotonic_time(:microsecond) - start_time
    updated_stats = update_stats(state.stats, :delete, table_type, duration, result == :ok)

    :telemetry.execute(
      [:protein_api, :ets, :delete],
      %{duration: duration},
      %{table_type: table_type, shard: shard_index, success: result == :ok}
    )

    {:reply, result, %{state | stats: updated_stats}}
  end

  def handle_call(:get_stats, _from, state) do
    enriched_stats = Map.merge(state.stats, %{
      table_sizes: get_table_sizes(state),
      memory_usage: get_memory_usage(state)
    })

    {:reply, enriched_stats, state}
  end

  def handle_call(:get_table_info, _from, state) do
    info = %{
      table_config: state.table_config,
      shard_count: state.shard_count,
      read_tables: Map.keys(state.read_tables),
      write_tables: Map.keys(state.write_tables)
    }

    {:reply, info, state}
  end

  defp create_table_shards(table_configs) do
    read_tables = %{}
    write_tables = %{}

    Enum.reduce(table_configs, {read_tables, write_tables}, fn {table_type, config}, {read_acc, write_acc} ->
      shard_count = round(config.shard_count)

      # Create read-optimized shards
      read_shards = for i <- 0..(shard_count-1) do
        table_name = :"#{table_type}_read_#{i}"

        :ets.new(table_name, [
          :set,
          :public,
          :named_table,
          {:read_concurrency, config.read_concurrency},
          {:write_concurrency, false},
          {:compressed, config.compressed}
        ])

        {i, table_name}
      end |> Map.new()

      # Create write-optimized shards
      write_shards = for i <- 0..(shard_count-1) do
        table_name = :"#{table_type}_write_#{i}"

        :ets.new(table_name, [
          :set,
          :public,
          :named_table,
          {:read_concurrency, false},
          {:write_concurrency, config.write_concurrency},
          {:compressed, config.compressed}
        ])

        {i, table_name}
      end |> Map.new()

      updated_read = Map.put(read_acc, table_type, read_shards)
      updated_write = Map.put(write_acc, table_type, write_shards)

      {updated_read, updated_write}
    end)
  end

  defp shard_for_key(key, table_type, state) do
    hash = :erlang.phash2(key)
    shard_count = round(state.table_config[table_type].shard_count)
    rem(hash, shard_count)
  end

  defp get_read_table(table_type, shard_index, state) do
    state.read_tables[table_type][shard_index]
  end

  defp get_write_table(table_type, shard_index, state) do
    state.write_tables[table_type][shard_index]
  end

  defp initialize_stats do
    %{
      reads: %{total: 0, hits: 0, misses: 0, avg_duration: 0},
      writes: %{total: 0, success: 0, failures: 0, avg_duration: 0},
      deletes: %{total: 0, success: 0, failures: 0, avg_duration: 0}
    }
  end

  defp update_stats(stats, operation, _table_type, duration, success) do
    case operation do
      :read ->
        reads = stats.reads
        new_total = reads.total + 1
        new_hits = if success, do: reads.hits + 1, else: reads.hits
        new_misses = if success, do: reads.misses, else: reads.misses + 1
        new_avg = ((reads.avg_duration * reads.total) + duration) / new_total

        %{stats | reads: %{total: new_total, hits: new_hits, misses: new_misses, avg_duration: new_avg}}

      :write ->
        writes = stats.writes
        new_total = writes.total + 1
        new_success = if success, do: writes.success + 1, else: writes.success
        new_failures = if success, do: writes.failures, else: writes.failures + 1
        new_avg = ((writes.avg_duration * writes.total) + duration) / new_total

        %{stats | writes: %{total: new_total, success: new_success, failures: new_failures, avg_duration: new_avg}}

      :delete ->
        deletes = stats.deletes
        new_total = deletes.total + 1
        new_success = if success, do: deletes.success + 1, else: deletes.success
        new_failures = if success, do: deletes.failures, else: deletes.failures + 1
        new_avg = ((deletes.avg_duration * deletes.total) + duration) / new_total

        %{stats | deletes: %{total: new_total, success: new_success, failures: new_failures, avg_duration: new_avg}}
    end
  end

  defp get_table_sizes(state) do
    for {table_type, shards} <- state.read_tables, into: %{} do
      read_size = Enum.sum(for {_shard, table} <- shards, do: :ets.info(table, :size))
      write_size = Enum.sum(for {_shard, table} <- state.write_tables[table_type], do: :ets.info(table, :size))

      {table_type, %{read_entries: read_size, write_entries: write_size, total: read_size + write_size}}
    end
  end

  defp get_memory_usage(state) do
    for {table_type, shards} <- state.read_tables, into: %{} do
      read_memory = Enum.sum(for {_shard, table} <- shards, do: :ets.info(table, :memory) * :erlang.system_info(:wordsize))
      write_memory = Enum.sum(for {_shard, table} <- state.write_tables[table_type], do: :ets.info(table, :memory) * :erlang.system_info(:wordsize))

      {table_type, %{read_bytes: read_memory, write_bytes: write_memory, total_bytes: read_memory + write_memory}}
    end
  end
end

# =======================================================================


# =======================================================================
