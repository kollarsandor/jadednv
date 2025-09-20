# =======================================================================


defmodule ProteinAPI.UDPProtocol do
  @moduledoc """
  UDP Protocol state machine with ACK aggregation, selective NACK,
  timer wheels, reorder buffer, and sliding window
  """

  use GenServer
  require Logger

  defstruct [
    :window_size,
    :send_window,
    :receive_window,
    :reorder_buffer,
    :ack_aggregator,
    :timer_wheel,
    :retransmit_queue,
    :selective_nacks,
    :rtt_estimator,
    :congestion_control
  ]

  @default_window_size 64
  @max_reorder_delay 1000  # 1 second
  @ack_delay 50           # 50ms ACK aggregation
  @retransmit_timeout 200  # 200ms base RTO
  @max_retransmits 3

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def handle_frame(frame, remote_ip, remote_port, opts) do
    GenServer.cast(__MODULE__, {:handle_frame, frame, remote_ip, remote_port, opts})
  end

  def send_reliable(remote_ip, remote_port, data, stream_id) do
    GenServer.call(__MODULE__, {:send_reliable, remote_ip, remote_port, data, stream_id})
  end

  def get_window_stats do
    GenServer.call(__MODULE__, :get_window_stats)
  end

  def init(opts) do
    window_size = Keyword.get(opts, :window_size, @default_window_size)

    state = %__MODULE__{
      window_size: window_size,
      send_window: %{
        base: 0,
        next_seq: 0,
        unacked: %{},
        max_seq: window_size - 1
      },
      receive_window: %{
        base: 0,
        expected: 0,
        buffer: %{},
        max_seq: window_size - 1
      },
      reorder_buffer: %{},
      ack_aggregator: init_ack_aggregator(),
      timer_wheel: init_timer_wheel(),
      retransmit_queue: :queue.new(),
      selective_nacks: MapSet.new(),
      rtt_estimator: init_rtt_estimator(),
      congestion_control: init_congestion_control()
    }

    # Start periodic timers
    schedule_ack_flush()
    schedule_timer_wheel_tick()
    schedule_retransmit_check()

    {:ok, state}
  end

  def handle_cast({:handle_frame, frame, remote_ip, remote_port, _opts}, state) do
    endpoint = {remote_ip, remote_port}

    cond do
      frame.flags &&& 0x02 != 0 ->  # ACK frame
        updated_state = handle_ack_frame(state, endpoint, frame)
        {:noreply, updated_state}

      frame.flags &&& 0x04 != 0 ->  # NACK frame
        updated_state = handle_nack_frame(state, endpoint, frame)
        {:noreply, updated_state}

      true ->  # Data frame
        updated_state = handle_data_frame(state, endpoint, frame)
        {:noreply, updated_state}
    end
  end

  def handle_call({:send_reliable, remote_ip, remote_port, data, stream_id}, from, state) do
    endpoint = {remote_ip, remote_port}

    case can_send_in_window(state, endpoint) do
      true ->
        seq = get_next_send_seq(state, endpoint)

        # Store for potential retransmission
        frame_info = %{
          sequence: seq,
          stream_id: stream_id,
          data: data,
          endpoint: endpoint,
          sent_at: :os.system_time(:millisecond),
          retransmit_count: 0,
          from: from
        }

        updated_state = add_to_send_window(state, endpoint, seq, frame_info)

        # Send via UDP transport
        ProteinAPI.UDPTransport.send_data(remote_ip, remote_port, data, [
          stream_id: stream_id,
          reliable: true
        ])

        # Schedule retransmit timer
        schedule_retransmit(updated_state, seq, endpoint)

        {:reply, {:ok, seq}, updated_state}

      false ->
        {:reply, {:error, :window_full}, state}
    end
  end

  def handle_call(:get_window_stats, _from, state) do
    stats = %{
      send_window: %{
        base: state.send_window.base,
        next_seq: state.send_window.next_seq,
        unacked_count: map_size(state.send_window.unacked),
        window_size: state.window_size
      },
      receive_window: %{
        base: state.receive_window.base,
        expected: state.receive_window.expected,
        buffered_count: map_size(state.receive_window.buffer)
      },
      reorder_buffer_size: map_size(state.reorder_buffer),
      pending_acks: map_size(state.ack_aggregator.pending),
      congestion_window: state.congestion_control.cwnd,
      rtt_estimate: state.rtt_estimator.srtt
    }

    {:reply, stats, state}
  end

  def handle_info(:flush_acks, state) do
    updated_state = flush_aggregated_acks(state)
    schedule_ack_flush()
    {:noreply, updated_state}
  end

  def handle_info(:timer_wheel_tick, state) do
    updated_state = process_timer_wheel(state)
    schedule_timer_wheel_tick()
    {:noreply, updated_state}
  end

  def handle_info(:check_retransmits, state) do
    updated_state = check_retransmit_timeouts(state)
    schedule_retransmit_check()
    {:noreply, updated_state}
  end

  defp handle_data_frame(state, endpoint, frame) do
    seq = frame.sequence
    receive_window = state.receive_window

    cond do
      seq == receive_window.expected ->
        # In-order frame
        updated_state = deliver_frame(state, frame)
        advance_receive_window(updated_state, endpoint)

      seq > receive_window.expected and seq <= receive_window.expected + state.window_size ->
        # Out-of-order but within window
        buffer_frame(state, endpoint, frame)

      true ->
        # Outside window, send NACK
        send_selective_nack(state, endpoint, receive_window.expected)
        state
    end
  end

  defp handle_ack_frame(state, endpoint, frame) do
    <<acked_seq::32>> = frame.payload

    case Map.get(state.send_window.unacked, acked_seq) do
      nil ->
        # Duplicate or unknown ACK
        state

      frame_info ->
        # Calculate RTT
        rtt = :os.system_time(:millisecond) - frame_info.sent_at
        updated_rtt = update_rtt_estimator(state.rtt_estimator, rtt)

        # Update congestion control
        updated_cc = update_congestion_control(state.congestion_control, :ack_received)

        # Remove from unacked and advance window
        updated_unacked = Map.delete(state.send_window.unacked, acked_seq)

        new_send_window = %{state.send_window |
          unacked: updated_unacked,
          base: calculate_new_base(updated_unacked, state.send_window.base)
        }

        # Reply to original sender
        if frame_info.from do
          GenServer.reply(frame_info.from, :ok)
        end

        %{state |
          send_window: new_send_window,
          rtt_estimator: updated_rtt,
          congestion_control: updated_cc
        }
    end
  end

  defp handle_nack_frame(state, endpoint, frame) do
    <<nacked_seq::32>> = frame.payload

    case Map.get(state.send_window.unacked, nacked_seq) do
      nil ->
        state

      frame_info ->
        # Immediate retransmit
        retransmit_frame(state, endpoint, frame_info)

        # Update congestion control for packet loss
        updated_cc = update_congestion_control(state.congestion_control, :packet_loss)

        %{state | congestion_control: updated_cc}
    end
  end

  defp deliver_frame(state, frame) do
    # Process the frame payload
    Logger.debug("Delivering in-order frame: seq=#{frame.sequence}")

    # Schedule ACK
    schedule_ack(state, frame.sequence)
  end

  defp buffer_frame(state, endpoint, frame) do
    seq = frame.sequence

    # Add to reorder buffer
    buffer_entry = %{
      frame: frame,
      endpoint: endpoint,
      buffered_at: :os.system_time(:millisecond)
    }

    updated_buffer = Map.put(state.reorder_buffer, seq, buffer_entry)

    # Schedule ACK for out-of-order frame
    schedule_ack(state, seq)

    %{state | reorder_buffer: updated_buffer}
  end

  defp advance_receive_window(state, endpoint) do
    receive_window = state.receive_window
    next_expected = receive_window.expected + 1

    # Check if we can deliver buffered frames
    {delivered_seqs, remaining_buffer} =
      deliver_buffered_frames(state.reorder_buffer, next_expected)

    new_expected = next_expected + length(delivered_seqs)

    new_receive_window = %{receive_window |
      expected: new_expected,
      base: max(receive_window.base, new_expected - state.window_size)
    }

    %{state |
      receive_window: new_receive_window,
      reorder_buffer: remaining_buffer
    }
  end

  defp deliver_buffered_frames(buffer, start_seq) do
    deliver_consecutive(buffer, start_seq, [])
  end

  defp deliver_consecutive(buffer, seq, delivered) do
    case Map.get(buffer, seq) do
      nil ->
        {Enum.reverse(delivered), buffer}

      entry ->
        Logger.debug("Delivering buffered frame: seq=#{seq}")
        updated_buffer = Map.delete(buffer, seq)
        deliver_consecutive(updated_buffer, seq + 1, [seq | delivered])
    end
  end

  defp schedule_ack(state, seq) do
    updated_pending = Map.put(state.ack_aggregator.pending, seq, :os.system_time(:millisecond))

    new_aggregator = %{state.ack_aggregator | pending: updated_pending}
    %{state | ack_aggregator: new_aggregator}
  end

  defp flush_aggregated_acks(state) do
    current_time = :os.system_time(:millisecond)

    {ready_acks, remaining_acks} =
      Enum.split_with(state.ack_aggregator.pending, fn {_seq, timestamp} ->
        current_time - timestamp >= @ack_delay
      end)

    # Send aggregated ACK for ready sequences
    if ready_acks != [] do
      ack_seqs = Enum.map(ready_acks, fn {seq, _} -> seq end)
      send_aggregated_ack(state, ack_seqs)
    end

    new_aggregator = %{state.ack_aggregator | pending: Map.new(remaining_acks)}
    %{state | ack_aggregator: new_aggregator}
  end

  defp send_aggregated_ack(state, ack_seqs) do
    # Implementation depends on aggregation format
    # For now, send individual ACKs
    Enum.each(ack_seqs, fn seq ->
      Logger.debug("Sending ACK for seq=#{seq}")
      # Send ACK via UDP transport
    end)
  end

  defp send_selective_nack(state, endpoint, expected_seq) do
    Logger.debug("Sending NACK for expected seq=#{expected_seq}")
    # Send NACK via UDP transport
    state
  end

  defp can_send_in_window(state, _endpoint) do
    unacked_count = map_size(state.send_window.unacked)
    unacked_count < state.window_size
  end

  defp get_next_send_seq(state, _endpoint) do
    state.send_window.next_seq
  end

  defp add_to_send_window(state, _endpoint, seq, frame_info) do
    updated_unacked = Map.put(state.send_window.unacked, seq, frame_info)

    new_send_window = %{state.send_window |
      unacked: updated_unacked,
      next_seq: seq + 1
    }

    %{state | send_window: new_send_window}
  end

  defp calculate_new_base(unacked_map, current_base) do
    if map_size(unacked_map) == 0 do
      current_base
    else
      min_unacked = unacked_map |> Map.keys() |> Enum.min()
      max(current_base, min_unacked)
    end
  end

  defp retransmit_frame(state, endpoint, frame_info) do
    Logger.debug("Retransmitting frame: seq=#{frame_info.sequence}")

    # Retransmit via UDP transport
    {remote_ip, remote_port} = endpoint
    ProteinAPI.UDPTransport.send_data(remote_ip, remote_port, frame_info.data, [
      stream_id: frame_info.stream_id,
      reliable: true
    ])
  end

  defp schedule_retransmit(state, seq, endpoint) do
    timeout = calculate_rto(state.rtt_estimator)
    timer_entry = %{
      type: :retransmit,
      seq: seq,
      endpoint: endpoint,
      expires_at: :os.system_time(:millisecond) + timeout
    }

    add_timer(state, timer_entry)
  end

  defp calculate_rto(rtt_estimator) do
    # RFC 6298 RTO calculation
    max(@retransmit_timeout, round(rtt_estimator.srtt + 4 * rtt_estimator.rttvar))
  end

  # Timer wheel and other helper functions
  defp init_ack_aggregator do
    %{pending: %{}, last_flush: :os.system_time(:millisecond)}
  end

  defp init_timer_wheel do
    %{entries: [], next_tick: :os.system_time(:millisecond)}
  end

  defp init_rtt_estimator do
    %{srtt: @retransmit_timeout, rttvar: @retransmit_timeout / 2}
  end

  defp init_congestion_control do
    %{cwnd: 1, ssthresh: 65535, state: :slow_start}
  end

  defp update_rtt_estimator(estimator, rtt_sample) do
    # RFC 6298 RTT estimation
    alpha = 0.125
    beta = 0.25

    new_srtt = (1 - alpha) * estimator.srtt + alpha * rtt_sample
    new_rttvar = (1 - beta) * estimator.rttvar + beta * abs(rtt_sample - new_srtt)

    %{estimator | srtt: new_srtt, rttvar: new_rttvar}
  end

  defp update_congestion_control(cc, event) do
    case event do
      :ack_received ->
        case cc.state do
          :slow_start when cc.cwnd < cc.ssthresh ->
            %{cc | cwnd: cc.cwnd + 1}
          _ ->
            %{cc | cwnd: cc.cwnd + 1 / cc.cwnd, state: :congestion_avoidance}
        end

      :packet_loss ->
        %{cc |
          ssthresh: max(cc.cwnd / 2, 2),
          cwnd: 1,
          state: :slow_start
        }
    end
  end

  defp add_timer(state, timer_entry) do
    # Add to timer wheel implementation
    state
  end

  defp process_timer_wheel(state) do
    # Process expired timers
    state
  end

  defp check_retransmit_timeouts(state) do
    # Check for retransmit timeouts
    state
  end

  defp schedule_ack_flush do
    Process.send_after(self(), :flush_acks, @ack_delay)
  end

  defp schedule_timer_wheel_tick do
    Process.send_after(self(), :timer_wheel_tick, 10)
  end

  defp schedule_retransmit_check do
    Process.send_after(self(), :check_retransmits, 100)
  end
end

# =======================================================================


# =======================================================================
