# =======================================================================


defmodule ProteinAPI.UDPTransport do
  @moduledoc """
  UDP Transport GenServer with MTU-aware framing, heartbeat, and statistics
  Frame format: version(1) | flags(1) | seq(4) | stream_id(4) | frag_id(2) | total_frags(2) | payload_size(2) | crc32(4) | payload(1200-1400)
  """

  use GenServer
  require Logger

  defstruct [
    :socket,
    :local_port,
    :remote_endpoints,
    :frame_size,
    :sequence_number,
    :heartbeat_interval,
    :stats,
    :protocol_handler
  ]

  @version 1
  @max_payload_size 1400
  @min_payload_size 1200
  @header_size 20  # version + flags + seq + stream_id + frag_id + total_frags + payload_size + crc32
  @heartbeat_interval 5000
  @heartbeat_timeout 15000

  # Frame flags
  @flag_heartbeat   0x01
  @flag_ack         0x02
  @flag_nack        0x04
  @flag_fragmented  0x08
  @flag_last_frag   0x10
  @flag_reliable    0x20

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def send_data(remote_ip, remote_port, data, opts \\ []) do
    GenServer.call(__MODULE__, {:send_data, remote_ip, remote_port, data, opts})
  end

  def add_remote_endpoint(ip, port, endpoint_id) do
    GenServer.call(__MODULE__, {:add_endpoint, ip, port, endpoint_id})
  end

  def get_stats do
    GenServer.call(__MODULE__, :get_stats)
  end

  def get_endpoint_health(endpoint_id) do
    GenServer.call(__MODULE__, {:get_endpoint_health, endpoint_id})
  end

  def init(opts) do
    local_port = Keyword.get(opts, :port, 0)
    frame_size = Keyword.get(opts, :frame_size, @max_payload_size)

    case :gen_udp.open(local_port, [:binary, :inet, {:active, true}, {:reuseaddr, true}]) do
      {:ok, socket} ->
        {:ok, actual_port} = :inet.port(socket)

        state = %__MODULE__{
          socket: socket,
          local_port: actual_port,
          remote_endpoints: %{},
          frame_size: min(frame_size, @max_payload_size),
          sequence_number: 0,
          heartbeat_interval: @heartbeat_interval,
          stats: init_stats(),
          protocol_handler: {ProteinAPI.UDPProtocol, []}
        }

        Logger.info("UDP Transport started on port #{actual_port}")
        schedule_heartbeat()

        {:ok, state}

      {:error, reason} ->
        Logger.error("Failed to open UDP socket: #{inspect(reason)}")
        {:stop, reason}
    end
  end

  def handle_call({:send_data, remote_ip, remote_port, data, opts}, from, state) do
    stream_id = Keyword.get(opts, :stream_id, :rand.uniform(0xFFFFFFFF))
    reliable = Keyword.get(opts, :reliable, false)

    case fragment_and_send(state, remote_ip, remote_port, data, stream_id, reliable) do
      :ok ->
        update_stats(state, :bytes_sent, byte_size(data))
        {:reply, :ok, state}

      {:error, reason} ->
        update_stats(state, :send_errors, 1)
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:add_endpoint, ip, port, endpoint_id}, _from, state) do
    endpoint = %{
      ip: ip,
      port: port,
      last_heartbeat: :os.system_time(:millisecond),
      rtt: 0,
      packet_loss: 0.0,
      status: :active
    }

    updated_endpoints = Map.put(state.remote_endpoints, endpoint_id, endpoint)
    new_state = %{state | remote_endpoints: updated_endpoints}

    Logger.info("Added UDP endpoint: #{endpoint_id} -> #{:inet.ntoa(ip)}:#{port}")
    {:reply, :ok, new_state}
  end

  def handle_call(:get_stats, _from, state) do
    {:reply, state.stats, state}
  end

  def handle_call({:get_endpoint_health, endpoint_id}, _from, state) do
    case Map.get(state.remote_endpoints, endpoint_id) do
      nil ->
        {:reply, {:error, :not_found}, state}
      endpoint ->
        health = calculate_endpoint_health(endpoint)
        {:reply, {:ok, health}, state}
    end
  end

  def handle_info({:udp, socket, remote_ip, remote_port, packet}, %{socket: socket} = state) do
    case parse_frame(packet) do
      {:ok, frame} ->
        updated_state = handle_received_frame(state, remote_ip, remote_port, frame)
        update_stats(updated_state, :packets_received, 1)
        {:noreply, updated_state}

      {:error, reason} ->
        Logger.warn("Invalid UDP frame from #{:inet.ntoa(remote_ip)}:#{remote_port}: #{reason}")
        update_stats(state, :invalid_frames, 1)
        {:noreply, state}
    end
  end

  def handle_info(:heartbeat, state) do
    send_heartbeats(state)
    check_endpoint_timeouts(state)
    schedule_heartbeat()
    {:noreply, state}
  end

  defp fragment_and_send(state, remote_ip, remote_port, data, stream_id, reliable) do
    payload_size = state.frame_size - @header_size
    data_size = byte_size(data)

    if data_size <= payload_size do
      # Single frame
      frame = create_frame(state, data, stream_id, 0, 1, reliable)
      send_frame(state, remote_ip, remote_port, frame)
    else
      # Multiple fragments
      fragments = create_fragments(data, payload_size)
      total_frags = length(fragments)

      Enum.with_index(fragments)
      |> Enum.reduce_while(:ok, fn {fragment, index}, _acc ->
        flags = if reliable, do: @flag_reliable, else: 0
        flags = flags ||| @flag_fragmented
        flags = if index == total_frags - 1, do: flags ||| @flag_last_frag, else: flags

        frame = create_frame(state, fragment, stream_id, index, total_frags, reliable, flags)

        case send_frame(state, remote_ip, remote_port, frame) do
          :ok -> {:cont, :ok}
          error -> {:halt, error}
        end
      end)
    end
  end

  defp create_frame(state, payload, stream_id, frag_id, total_frags, reliable, custom_flags \\ nil) do
    seq = get_next_sequence(state)
    payload_size = byte_size(payload)

    flags = custom_flags || begin
      base_flags = if reliable, do: @flag_reliable, else: 0
      if total_frags > 1 do
        base_flags ||| @flag_fragmented
      else
        base_flags
      end
    end

    header = <<
      @version::8,
      flags::8,
      seq::32,
      stream_id::32,
      frag_id::16,
      total_frags::16,
      payload_size::16
    >>

    frame_without_crc = header <> payload
    crc = :erlang.crc32(frame_without_crc)

    frame_without_crc <> <<crc::32>>
  end

  defp create_fragments(data, max_size) do
    data
    |> :binary.bin_to_list()
    |> Enum.chunk_every(max_size)
    |> Enum.map(&:binary.list_to_bin/1)
  end

  defp send_frame(state, remote_ip, remote_port, frame) do
    case :gen_udp.send(state.socket, remote_ip, remote_port, frame) do
      :ok ->
        update_stats(state, :packets_sent, 1)
        :ok
      error ->
        Logger.error("Failed to send UDP frame: #{inspect(error)}")
        error
    end
  end

  defp parse_frame(packet) when byte_size(packet) < @header_size do
    {:error, :packet_too_small}
  end

  defp parse_frame(packet) do
    case packet do
      <<version::8, flags::8, seq::32, stream_id::32, frag_id::16,
        total_frags::16, payload_size::16, payload::binary-size(payload_size),
        crc::32, _rest::binary>> when version == @version ->

        frame_without_crc = binary_part(packet, 0, byte_size(packet) - 4)

        if :erlang.crc32(frame_without_crc) == crc do
          {:ok, %{
            version: version,
            flags: flags,
            sequence: seq,
            stream_id: stream_id,
            fragment_id: frag_id,
            total_fragments: total_frags,
            payload_size: payload_size,
            payload: payload,
            crc: crc
          }}
        else
          {:error, :invalid_crc}
        end

      <<version::8, _rest::binary>> when version != @version ->
        {:error, :unsupported_version}

      _ ->
        {:error, :malformed_packet}
    end
  end

  defp handle_received_frame(state, remote_ip, remote_port, frame) do
    cond do
      frame.flags &&& @flag_heartbeat != 0 ->
        handle_heartbeat(state, remote_ip, remote_port, frame)

      frame.flags &&& @flag_ack != 0 ->
        handle_ack(state, remote_ip, remote_port, frame)

      frame.flags &&& @flag_nack != 0 ->
        handle_nack(state, remote_ip, remote_port, frame)

      true ->
        handle_data_frame(state, remote_ip, remote_port, frame)
    end
  end

  defp handle_heartbeat(state, remote_ip, remote_port, frame) do
    endpoint_key = {remote_ip, remote_port}

    # Send heartbeat response
    response_frame = create_heartbeat_response(state)
    send_frame(state, remote_ip, remote_port, response_frame)

    # Update endpoint info
    update_endpoint_heartbeat(state, endpoint_key)
  end

  defp handle_data_frame(state, remote_ip, remote_port, frame) do
    {protocol_module, protocol_opts} = state.protocol_handler

    # Delegate to protocol handler
    protocol_module.handle_frame(frame, remote_ip, remote_port, protocol_opts)

    # Send ACK if reliable delivery requested
    if frame.flags &&& @flag_reliable != 0 do
      ack_frame = create_ack_frame(state, frame.sequence, frame.stream_id)
      send_frame(state, remote_ip, remote_port, ack_frame)
    end

    state
  end

  defp handle_ack(state, _remote_ip, _remote_port, frame) do
    Logger.debug("Received ACK for sequence #{frame.sequence}")
    state
  end

  defp handle_nack(state, remote_ip, remote_port, frame) do
    Logger.warn("Received NACK for sequence #{frame.sequence}")
    # Implement retransmission logic here
    state
  end

  defp create_heartbeat_response(state) do
    create_frame(state, <<>>, 0, 0, 1, false, @flag_heartbeat)
  end

  defp create_ack_frame(state, ack_seq, stream_id) do
    create_frame(state, <<ack_seq::32>>, stream_id, 0, 1, false, @flag_ack)
  end

  defp send_heartbeats(state) do
    Enum.each(state.remote_endpoints, fn {_id, endpoint} ->
      heartbeat_frame = create_frame(state, <<>>, 0, 0, 1, false, @flag_heartbeat)
      send_frame(state, endpoint.ip, endpoint.port, heartbeat_frame)
    end)
  end

  defp check_endpoint_timeouts(state) do
    current_time = :os.system_time(:millisecond)

    Enum.each(state.remote_endpoints, fn {endpoint_id, endpoint} ->
      if current_time - endpoint.last_heartbeat > @heartbeat_timeout do
        Logger.warn("Endpoint #{endpoint_id} heartbeat timeout")
        ProteinAPI.HealthController.report_degradation(:udp_transport, endpoint_id, :heartbeat_timeout)
      end
    end)
  end

  defp update_endpoint_heartbeat(state, endpoint_key) do
    # Implementation for updating endpoint heartbeat timestamp
    state
  end

  defp calculate_endpoint_health(endpoint) do
    current_time = :os.system_time(:millisecond)
    time_since_heartbeat = current_time - endpoint.last_heartbeat

    %{
      status: endpoint.status,
      rtt: endpoint.rtt,
      packet_loss: endpoint.packet_loss,
      last_heartbeat: endpoint.last_heartbeat,
      time_since_heartbeat: time_since_heartbeat,
      healthy: time_since_heartbeat < @heartbeat_timeout
    }
  end

  defp get_next_sequence(state) do
    # Atomic sequence number increment
    :atomics.add_get(state.sequence_number, 1, 1)
  end

  defp schedule_heartbeat do
    Process.send_after(self(), :heartbeat, @heartbeat_interval)
  end

  defp init_stats do
    %{
      packets_sent: 0,
      packets_received: 0,
      bytes_sent: 0,
      bytes_received: 0,
      send_errors: 0,
      invalid_frames: 0,
      started_at: :os.system_time(:millisecond)
    }
  end

  defp update_stats(state, metric, value) do
    Map.update!(state.stats, metric, &(&1 + value))
  end
end

# =======================================================================


# =======================================================================
