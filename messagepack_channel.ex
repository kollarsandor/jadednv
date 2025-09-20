# =======================================================================


defmodule ProteinAPI.MessagePackChannel do
  @moduledoc """
  JSON → MessagePack channel Phoenix ⇄ JuliaPort (Msgpax)
  High-performance binary serialization for reduced bandwidth and faster parsing
  """

  use GenServer
  require Logger

  defstruct [
    :julia_port,
    :pending_requests,
    :request_counter,
    :encoding_stats,
    :compression_enabled
  ]

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(opts) do
    state = %__MODULE__{
      julia_port: nil,
      pending_requests: %{},
      request_counter: 0,
      encoding_stats: %{
        json_bytes: 0,
        msgpack_bytes: 0,
        compression_ratio: 0.0,
        encoding_time_ms: 0
      },
      compression_enabled: Keyword.get(opts, :compression, true)
    }

    {:ok, state, {:continue, :start_julia_port}}
  end

  def handle_continue(:start_julia_port, state) do
    case start_julia_port() do
      {:ok, port} ->
        {:noreply, %{state | julia_port: port}}
      {:error, reason} ->
        Logger.error("Failed to start Julia port: #{inspect(reason)}")
        {:stop, reason, state}
    end
  end

  def send_command(command, data, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 30_000)
    encoding = Keyword.get(opts, :encoding, :auto)
    compression = Keyword.get(opts, :compression, true)

    GenServer.call(__MODULE__, {:send_command, command, data, encoding, compression}, timeout)
  end

  def get_encoding_stats do
    GenServer.call(__MODULE__, :get_encoding_stats)
  end

  def handle_call({:send_command, command, data, encoding, compression}, from, state) do
    request_id = state.request_counter + 1

    # Determine optimal encoding
    optimal_encoding = determine_encoding(data, encoding)

    # Encode the request
    {encoded_data, encoding_time, encoded_size} = encode_request(command, data, optimal_encoding, compression)

    # Update encoding statistics
    updated_stats = update_encoding_stats(state.encoding_stats, data, encoded_data, encoding_time, optimal_encoding)

    # Send to Julia port
    case send_to_julia_port(state.julia_port, encoded_data) do
      :ok ->
        updated_pending = Map.put(state.pending_requests, request_id, {from, System.monotonic_time(:millisecond)})

        :telemetry.execute(
          [:protein_api, :julia_port, :command_sent],
          %{
            payload_size: encoded_size,
            encoding_time: encoding_time
          },
          %{
            command_type: command,
            request_id: request_id,
            encoding: optimal_encoding,
            compression: compression
          }
        )

        updated_state = %{state |
          request_counter: request_id,
          pending_requests: updated_pending,
          encoding_stats: updated_stats
        }

        {:noreply, updated_state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call(:get_encoding_stats, _from, state) do
    {:reply, state.encoding_stats, state}
  end

  def handle_info({port, {:data, data}}, %{julia_port: port} = state) when is_port(port) do
    case decode_response(data) do
      {:ok, request_id, response, decoding_time} ->
        case Map.pop(state.pending_requests, request_id) do
          {{from, start_time}, updated_pending} ->
            total_time = System.monotonic_time(:millisecond) - start_time

            :telemetry.execute(
              [:protein_api, :julia_port, :response_received],
              %{
                duration: total_time * 1000, # Convert to microseconds
                response_size: byte_size(data),
                decoding_time: decoding_time
              },
              %{
                request_id: request_id,
                command_type: "unknown" # Would need to track this
              }
            )

            GenServer.reply(from, {:ok, response})
            {:noreply, %{state | pending_requests: updated_pending}}

          {nil, _} ->
            Logger.warn("Received response for unknown request_id: #{request_id}")
            {:noreply, state}
        end

      {:error, reason} ->
        Logger.error("Failed to decode Julia response: #{inspect(reason)}")

        :telemetry.execute(
          [:protein_api, :julia_port, :error],
          %{},
          %{
            error: reason,
            error_type: "decode_error",
            command_type: "unknown"
          }
        )

        {:noreply, state}
    end
  end

  def handle_info({port, {:exit_status, status}}, %{julia_port: port} = state) do
    Logger.error("Julia port exited with status: #{status}")

    # Fail all pending requests
    Enum.each(state.pending_requests, fn {_request_id, {from, _start_time}} ->
      GenServer.reply(from, {:error, :julia_port_died})
    end)

    {:stop, :julia_port_died, state}
  end

  defp start_julia_port do
    julia_script = Path.join([File.cwd!(), "julia_messagepack_server.jl"])

    case Port.open({:spawn, "julia #{julia_script}"},
      [:binary, :exit_status, packet: 4]) do
      port when is_port(port) ->
        {:ok, port}
      error ->
        {:error, error}
    end
  end

  defp determine_encoding(data, :auto) do
    # Automatically determine the best encoding based on data characteristics
    json_size = byte_size(Jason.encode!(data))

    case Msgpax.pack(data) do
      {:ok, msgpack_data} ->
        msgpack_size = byte_size(msgpack_data)

        # Choose MessagePack if it's significantly smaller or for complex structures
        if msgpack_size < json_size * 0.8 or has_binary_data?(data) do
          :messagepack
        else
          :json
        end

      {:error, _} ->
        # Fall back to JSON if MessagePack fails
        :json
    end
  end
  defp determine_encoding(_data, encoding), do: encoding

  defp has_binary_data?(data) when is_map(data) do
    Enum.any?(data, fn
      {_k, v} when is_binary(v) -> byte_size(v) > 1000
      {_k, v} -> has_binary_data?(v)
    end)
  end
  defp has_binary_data?(data) when is_list(data) do
    Enum.any?(data, &has_binary_data?/1)
  end
  defp has_binary_data?(_), do: false

  defp encode_request(command, data, encoding, compression) do
    start_time = System.monotonic_time(:microsecond)

    request = %{
      command: command,
      data: data,
      encoding: encoding,
      compression: compression,
      timestamp: System.system_time(:microsecond)
    }

    encoded = case encoding do
      :messagepack ->
        case Msgpax.pack(request) do
          {:ok, packed} -> packed
          {:error, _} ->
            # Fall back to JSON if MessagePack fails
            Jason.encode!(request)
        end

      :json ->
        Jason.encode!(request)
    end

    # Apply compression if enabled
    final_data = if compression do
      :zlib.compress(encoded)
    else
      encoded
    end

    encoding_time = System.monotonic_time(:microsecond) - start_time

    {final_data, encoding_time, byte_size(final_data)}
  end

  defp decode_response(data) do
    start_time = System.monotonic_time(:microsecond)

    try do
      # Try to decompress first
      decompressed = try do
        :zlib.uncompress(data)
      rescue
        _ -> data
      end

      # Try MessagePack first, then JSON
      response = case Msgpax.unpack(decompressed) do
        {:ok, unpacked} -> unpacked
        {:error, _} ->
          case Jason.decode(decompressed) do
            {:ok, decoded} -> decoded
            {:error, _} -> throw(:decode_error)
          end
      end

      decoding_time = System.monotonic_time(:microsecond) - start_time

      request_id = Map.get(response, "request_id") || Map.get(response, :request_id)
      result = Map.get(response, "result") || Map.get(response, :result)

      {:ok, request_id, result, decoding_time}

    catch
      _ -> {:error, :decode_failed}
    end
  end

  defp send_to_julia_port(port, data) do
    try do
      Port.command(port, data)
      :ok
    catch
      :error, reason -> {:error, reason}
    end
  end

  defp update_encoding_stats(stats, original_data, encoded_data, encoding_time, encoding) do
    json_size = byte_size(Jason.encode!(original_data))
    encoded_size = byte_size(encoded_data)

    total_json_bytes = stats.json_bytes + json_size
    total_encoded_bytes = case encoding do
      :messagepack -> stats.msgpack_bytes + encoded_size
      :json -> stats.msgpack_bytes
    end

    new_compression_ratio = if total_json_bytes > 0 do
      total_encoded_bytes / total_json_bytes
    else
      0.0
    end

    %{
      json_bytes: total_json_bytes,
      msgpack_bytes: total_encoded_bytes,
      compression_ratio: new_compression_ratio,
      encoding_time_ms: stats.encoding_time_ms + (encoding_time / 1000)
    }
  end
end

defmodule ProteinAPI.MessagePackChannel.Benchmark do
  @moduledoc """
  Benchmarking tools for MessagePack vs JSON performance
  """

  def compare_encodings(test_data) do
    # JSON encoding benchmark
    {json_time, json_result} = :timer.tc(fn ->
      Jason.encode!(test_data)
    end)

    # MessagePack encoding benchmark
    {msgpack_time, msgpack_result} = :timer.tc(fn ->
      case Msgpax.pack(test_data) do
        {:ok, packed} -> packed
        {:error, _} -> Jason.encode!(test_data)
      end
    end)

    # Compression benchmarks
    {json_compress_time, json_compressed} = :timer.tc(fn ->
      :zlib.compress(json_result)
    end)

    {msgpack_compress_time, msgpack_compressed} = :timer.tc(fn ->
      :zlib.compress(msgpack_result)
    end)

    %{
      json: %{
        encode_time_us: json_time,
        size_bytes: byte_size(json_result),
        compressed_size: byte_size(json_compressed),
        compress_time_us: json_compress_time,
        total_time_us: json_time + json_compress_time
      },
      messagepack: %{
        encode_time_us: msgpack_time,
        size_bytes: byte_size(msgpack_result),
        compressed_size: byte_size(msgpack_compressed),
        compress_time_us: msgpack_compress_time,
        total_time_us: msgpack_time + msgpack_compress_time
      },
      savings: %{
        size_reduction: (byte_size(json_result) - byte_size(msgpack_result)) / byte_size(json_result),
        compressed_size_reduction: (byte_size(json_compressed) - byte_size(msgpack_compressed)) / byte_size(json_compressed),
        time_improvement: (json_time + json_compress_time - msgpack_time - msgpack_compress_time) / (json_time + json_compress_time)
      }
    }
  end

  def benchmark_protein_data do
    # Generate realistic protein folding data
    test_structures = %{
      small_protein: generate_protein_structure(50),
      medium_protein: generate_protein_structure(200),
      large_protein: generate_protein_structure(1000),
      batch_request: %{
        sequences: Enum.map(1..10, fn _ -> generate_sequence(100) end),
        options: %{
          backend: "ibm_torino",
          shots: 1024,
          optimization_level: 3
        }
      },
      quantum_result: %{
        job_id: "quantum_#{:rand.uniform(10000)}",
        measurements: Enum.map(1..1024, fn _ -> :rand.uniform() end),
        quantum_state: %{
          amplitudes: Enum.map(1..64, fn _ ->
            %{real: :rand.normal(), imag: :rand.normal()}
          end)
        },
        error_mitigation: %{
          shots_used: 1024,
          error_rate: 0.001,
          fidelity: 0.99
        }
      }
    }

    results = Enum.map(test_structures, fn {name, data} ->
      benchmark_result = compare_encodings(data)
      {name, benchmark_result}
    end)

    IO.puts("\n=== MessagePack vs JSON Benchmark Results ===")

    Enum.each(results, fn {name, result} ->
      IO.puts("\n#{name}:")
      IO.puts("  JSON: #{result.json.size_bytes} bytes, #{result.json.encode_time_us}μs encode")
      IO.puts("  MessagePack: #{result.messagepack.size_bytes} bytes, #{result.messagepack.encode_time_us}μs encode")
      IO.puts("  Size reduction: #{Float.round(result.savings.size_reduction * 100, 1)}%")
      IO.puts("  Time improvement: #{Float.round(result.savings.time_improvement * 100, 1)}%")
    end)

    results
  end

  defp generate_protein_structure(num_residues) do
    %{
      atoms: Enum.map(1..(num_residues * 4), fn i ->
        %{
          id: i,
          element: Enum.random(["C", "N", "O", "S"]),
          x: :rand.uniform() * 100,
          y: :rand.uniform() * 100,
          z: :rand.uniform() * 100,
          occupancy: :rand.uniform(),
          b_factor: :rand.uniform() * 50
        }
      end),
      bonds: Enum.map(1..(num_residues * 3), fn i ->
        %{
          atom1: i,
          atom2: i + 1,
          order: Enum.random([1, 2, 3])
        }
      end),
      confidence: Enum.map(1..num_residues, fn _ -> :rand.uniform() end),
      energy: :rand.uniform() * -1000,
      metadata: %{
        method: "alphafold3",
        version: "3.0.0",
        timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
      }
    }
  end

  defp generate_sequence(length) do
    amino_acids = ~w(A R N D C Q E G H I L K M F P S T W Y V)
    Enum.map(1..length, fn _ -> Enum.random(amino_acids) end) |> Enum.join()
  end
end

# =======================================================================


# =======================================================================
