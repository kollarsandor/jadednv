# =======================================================================


defmodule ProteinAPI.RequestTracer do
  @moduledoc """
  Request-ID tracer chain: Phoenix → ComputationManager → JuliaPort → Julia
  Provides distributed tracing across the entire request lifecycle
  """

  use GenServer
  require Logger

  @trace_header "x-request-id"
  @trace_context_key :request_trace_context

  defstruct [
    :request_id,
    :parent_span_id,
    :trace_id,
    :start_time,
    :user_agent,
    :client_ip,
    :method,
    :path,
    :spans,
    :baggage
  ]

  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def init(_) do
    :ets.new(:request_traces, [:set, :public, :named_table, {:read_concurrency, true}])
    {:ok, %{}}
  end

  def start_trace(conn) do
    request_id = get_or_generate_request_id(conn)
    trace_id = generate_trace_id()

    trace_context = %__MODULE__{
      request_id: request_id,
      trace_id: trace_id,
      start_time: System.monotonic_time(:microsecond),
      user_agent: get_user_agent(conn),
      client_ip: get_client_ip(conn),
      method: conn.method,
      path: conn.request_path,
      spans: [],
      baggage: %{}
    }

    :ets.insert(:request_traces, {request_id, trace_context})

    # Set OpenTelemetry context
    :otel_tracer.set_current_span(
      :otel_tracer.start_span(:default, "http_request", %{
        "http.method" => conn.method,
        "http.url" => conn.request_path,
        "http.user_agent" => trace_context.user_agent,
        "http.client_ip" => trace_context.client_ip,
        "request.id" => request_id,
        "trace.id" => trace_id
      })
    )

    :telemetry.execute(
      [:protein_api, :request, :start],
      %{timestamp: System.system_time(:microsecond)},
      %{
        request_id: request_id,
        trace_id: trace_id,
        method: conn.method,
        route: conn.request_path
      }
    )

    conn
    |> Plug.Conn.put_private(@trace_context_key, trace_context)
    |> Plug.Conn.put_resp_header(@trace_header, request_id)
  end

  def end_trace(conn, status) do
    case Plug.Conn.get_private(conn, @trace_context_key) do
      nil -> conn
      trace_context ->
        duration = System.monotonic_time(:microsecond) - trace_context.start_time

        :otel_span.set_attributes([
          {"http.status_code", status},
          {"http.response_size", get_response_size(conn)}
        ])
        :otel_span.end_span()

        :telemetry.execute(
          [:protein_api, :request, :stop],
          %{
            duration: duration,
            response_size: get_response_size(conn)
          },
          %{
            request_id: trace_context.request_id,
            trace_id: trace_context.trace_id,
            method: trace_context.method,
            route: trace_context.path,
            status: status
          }
        )

        # Keep trace for a short time for debugging
        Process.send_after(self(), {:cleanup_trace, trace_context.request_id}, :timer.minutes(5))

        conn
    end
  end

  def add_span(request_id, span_name, metadata \\ %{}) do
    case :ets.lookup(:request_traces, request_id) do
      [{^request_id, trace_context}] ->
        span = %{
          id: generate_span_id(),
          name: span_name,
          start_time: System.monotonic_time(:microsecond),
          parent_id: trace_context.parent_span_id,
          metadata: metadata,
          end_time: nil
        }

        updated_context = %{trace_context |
          spans: [span | trace_context.spans],
          parent_span_id: span.id
        }

        :ets.insert(:request_traces, {request_id, updated_context})

        # Start OpenTelemetry span
        :otel_tracer.start_span(:default, span_name, Map.merge(metadata, %{
          "span.id" => span.id,
          "parent.span.id" => span.parent_id,
          "request.id" => request_id
        }))

        span.id
      [] ->
        Logger.warn("Trace not found for request_id: #{request_id}")
        nil
    end
  end

  def end_span(request_id, span_id, metadata \\ %{}) do
    case :ets.lookup(:request_traces, request_id) do
      [{^request_id, trace_context}] ->
        updated_spans = Enum.map(trace_context.spans, fn span ->
          if span.id == span_id do
            %{span |
              end_time: System.monotonic_time(:microsecond),
              metadata: Map.merge(span.metadata, metadata)
            }
          else
            span
          end
        end)

        updated_context = %{trace_context | spans: updated_spans}
        :ets.insert(:request_traces, {request_id, updated_context})

        # End OpenTelemetry span
        :otel_span.set_attributes(metadata)
        :otel_span.end_span()

      [] ->
        Logger.warn("Trace not found for request_id: #{request_id}")
    end
  end

  def get_trace(request_id) do
    case :ets.lookup(:request_traces, request_id) do
      [{^request_id, trace_context}] -> {:ok, trace_context}
      [] -> {:error, :not_found}
    end
  end

  def current_request_id do
    case Process.get(@trace_context_key) do
      nil -> nil
      trace_context -> trace_context.request_id
    end
  end

  def set_baggage(request_id, key, value) do
    case :ets.lookup(:request_traces, request_id) do
      [{^request_id, trace_context}] ->
        updated_baggage = Map.put(trace_context.baggage, key, value)
        updated_context = %{trace_context | baggage: updated_baggage}
        :ets.insert(:request_traces, {request_id, updated_context})
        :ok
      [] ->
        {:error, :not_found}
    end
  end

  def get_baggage(request_id, key) do
    case :ets.lookup(:request_traces, request_id) do
      [{^request_id, trace_context}] ->
        Map.get(trace_context.baggage, key)
      [] ->
        nil
    end
  end

  def handle_info({:cleanup_trace, request_id}, state) do
    :ets.delete(:request_traces, request_id)
    {:noreply, state}
  end

  defp get_or_generate_request_id(conn) do
    case Plug.Conn.get_req_header(conn, @trace_header) do
      [request_id] -> request_id
      [] -> generate_request_id()
    end
  end

  defp generate_request_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end

  defp generate_trace_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end

  defp generate_span_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp get_user_agent(conn) do
    case Plug.Conn.get_req_header(conn, "user-agent") do
      [user_agent] -> user_agent
      [] -> "unknown"
    end
  end

  defp get_client_ip(conn) do
    case Plug.Conn.get_req_header(conn, "x-forwarded-for") do
      [forwarded] ->
        forwarded |> String.split(",") |> List.first() |> String.trim()
      [] ->
        case conn.remote_ip do
          {a, b, c, d} -> "#{a}.#{b}.#{c}.#{d}"
          _ -> "unknown"
        end
    end
  end

  defp get_response_size(conn) do
    case Plug.Conn.get_resp_header(conn, "content-length") do
      [size] -> String.to_integer(size)
      [] -> 0
    end
  end
end

defmodule ProteinAPI.RequestTracer.Plug do
  @moduledoc """
  Plug for automatic request tracing
  """

  import Plug.Conn
  require Logger

  def init(opts), do: opts

  def call(conn, _opts) do
    conn = ProteinAPI.RequestTracer.start_trace(conn)

    register_before_send(conn, fn conn ->
      status = conn.status || 200
      ProteinAPI.RequestTracer.end_trace(conn, status)
    end)
  end
end

# =======================================================================


# =======================================================================
