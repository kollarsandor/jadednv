# =======================================================================


defmodule ProteinAPI.HTTP2Optimization do
  @moduledoc """
  HTTP/2 and keep-alive tuning for Phoenix Endpoint
  Optimizes connection handling and reduces latency
  """

  def configure_endpoint do
    %{
      # HTTP/2 configuration
      http: [
        ip: {0, 0, 0, 0},
        port: 4000,
        protocol_options: [
          # Enable HTTP/2
          enable_http2: true,

          # Connection settings
          max_connections: 16384,
          num_acceptors: 100,

          # Keep-alive settings
          idle_timeout: 60_000,
          request_timeout: 30_000,

          # HTTP/2 specific settings
          max_frame_size: 16384,
          max_header_list_size: 8192,
          initial_window_size: 65535,
          max_concurrent_streams: 1000,

          # Flow control
          enable_connect_protocol: true,

          # Compression
          compress: true,

          # Buffer sizes
          buffer_size: 32768,
          recbuf: 32768,
          sndbuf: 32768,

          # TCP options
          tcp_options: [
            :binary,
            {:packet, :raw},
            {:reuseaddr, true},
            {:keepalive, true},
            {:backlog, 1024},
            {:nodelay, true},
            {:send_timeout, 30_000},
            {:send_timeout_close, true}
          ]
        ]
      ],

      # HTTPS configuration with HTTP/2
      https: [
        ip: {0, 0, 0, 0},
        port: 4001,
        cipher_suite: :strong,
        keyfile: System.get_env("SSL_KEY_PATH"),
        certfile: System.get_env("SSL_CERT_PATH"),

        protocol_options: [
          # Enable HTTP/2 over TLS (h2)
          alpn_preferred_protocols: ["h2", "http/1.1"],

          # HTTP/2 settings
          enable_http2: true,
          max_frame_size: 16384,
          max_header_list_size: 8192,
          initial_window_size: 65535,
          max_concurrent_streams: 1000,

          # Connection pooling
          max_connections: 16384,
          num_acceptors: 100,

          # Timeouts
          idle_timeout: 60_000,
          request_timeout: 30_000,

          # Compression
          compress: true,

          # TLS-specific options
          honor_cipher_order: true,
          secure_renegotiate: true,
          reuse_sessions: true,

          tcp_options: [
            :binary,
            {:packet, :raw},
            {:reuseaddr, true},
            {:keepalive, true},
            {:backlog, 1024},
            {:nodelay, true},
            {:send_timeout, 30_000},
            {:send_timeout_close, true}
          ]
        ]
      ]
    }
  end

  def setup_connection_pool do
    # Configure Hackney HTTP client pool for outbound requests
    :hackney_pool.start_pool(:protein_api_pool, [
      timeout: 30_000,
      max_connections: 1000,
      retry: 3,
      retry_timeout: 5_000,
      follow_redirect: true,
      force_redirect: true,
      pool_size: 50,

      # HTTP/2 settings for outbound connections
      http_options: [
        version: :"HTTP/2",
        tcp_options: [
          :binary,
          {:keepalive, true},
          {:nodelay, true},
          {:send_timeout, 30_000},
          {:recv_timeout, 30_000}
        ]
      ]
    ])
  end

  def configure_cowboy_settings do
    # Advanced Cowboy configuration for HTTP/2 optimization
    %{
      env: %{
        dispatch: :cowboy_router.compile([
          {'_', [
            {"/api/[...]", ProteinAPIWeb.Endpoint, []},
            {"/health", ProteinAPI.HealthHandler, []},
            {"/metrics", ProteinAPI.MetricsHandler, []}
          ]}
        ])
      },

      # Protocol options
      stream_handlers: [:cowboy_compress_h, :cowboy_stream_h],

      # Connection settings
      connection_type: :supervisor,
      max_keepalive: 1000,

      # Request handling
      max_empty_lines: 5,
      max_header_name_length: 64,
      max_header_value_length: 4096,
      max_headers: 100,
      max_request_line_length: 8000,

      # Timeouts
      request_timeout: 30_000,
      idle_timeout: 60_000,

      # Flow control for HTTP/2
      initial_connection_window_size: 65535,
      initial_stream_window_size: 65535,
      max_concurrent_streams: 1000,

      # Frame size limits
      max_decode_table_size: 4096,
      max_encode_table_size: 4096,
      max_frame_size_received: 16384,
      max_frame_size_sent: 16384,

      # Settings frame
      settings_timeout: 5000,
      enable_connect_protocol: false,

      # Compression
      compress: true,
      deflate_options: %{
        level: 6,
        mem_level: 8,
        strategy: :default
      }
    }
  end

  def setup_middleware_optimization do
    [
      # Early request processing
      {ProteinAPI.EarlyRequestProcessor, []},

      # Connection keep-alive handler
      {ProteinAPI.KeepAliveHandler, []},

      # Request deduplication
      {ProteinAPI.RequestDeduplicator, []},

      # Response compression
      {Plug.Deflate, []},

      # CORS with preflight optimization
      {CORSPlug, [
        origin: ["*"],
        credentials: false,
        max_age: 86400,
        headers: ["Authorization", "Content-Type", "X-Request-ID"],
        methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
      ]},

      # Request ID tracking
      {ProteinAPI.RequestTracer.Plug, []},

      # Rate limiting with sliding window
      {ProteinAPI.RateLimiter, [
        max_requests: 1000,
        window_ms: 60_000,
        identifier: &get_client_identifier/1
      ]},

      # Request parsing with streaming
      {Plug.Parsers, [
        parsers: [:urlencoded, :multipart, {:json, json_decoder: Jason}],
        pass: ["*/*"],
        body_reader: {ProteinAPI.StreamingBodyReader, :read_body, []},
        length: 50_000_000, # 50MB limit
        read_length: 1_000_000,
        read_timeout: 30_000
      ]},

      # Method override
      {Plug.MethodOverride, []},

      # Head requests
      {Plug.Head, []},

      # Session handling with optimized storage
      {Plug.Session, [
        store: :ets,
        key: "_protein_api_session",
        table: :session_store,
        secure: true,
        http_only: true,
        same_site: "Lax",
        max_age: 86400
      ]}
    ]
  end

  defp get_client_identifier(conn) do
    case Plug.Conn.get_req_header(conn, "x-forwarded-for") do
      [forwarded] -> forwarded |> String.split(",") |> List.first() |> String.trim()
      [] -> to_string(:inet_parse.ntoa(conn.remote_ip))
    end
  end
end

defmodule ProteinAPI.EarlyRequestProcessor do
  @moduledoc """
  Early request processing for performance optimization
  """

  import Plug.Conn

  def init(opts), do: opts

  def call(conn, _opts) do
    start_time = System.monotonic_time(:microsecond)

    conn
    |> put_private(:request_start_time, start_time)
    |> maybe_handle_preflight()
    |> set_performance_headers()
    |> enable_http2_push()
  end

  defp maybe_handle_preflight(conn) do
    if conn.method == "OPTIONS" do
      conn
      |> put_resp_header("access-control-allow-origin", "*")
      |> put_resp_header("access-control-allow-methods", "GET, POST, PUT, DELETE, OPTIONS")
      |> put_resp_header("access-control-allow-headers", "authorization, content-type, x-request-id")
      |> put_resp_header("access-control-max-age", "86400")
      |> send_resp(200, "")
      |> halt()
    else
      conn
    end
  end

  defp set_performance_headers(conn) do
    conn
    |> put_resp_header("x-content-type-options", "nosniff")
    |> put_resp_header("x-frame-options", "DENY")
    |> put_resp_header("x-xss-protection", "1; mode=block")
    |> put_resp_header("connection", "keep-alive")
    |> put_resp_header("keep-alive", "timeout=60, max=1000")
  end

  defp enable_http2_push(conn) do
    # HTTP/2 Server Push for critical resources
    if get_req_header(conn, "upgrade") == ["h2c"] or
       get_req_header(conn, ":scheme") == ["https"] do
      conn
      |> put_resp_header("link", "</css/app.css>; rel=preload; as=style")
      |> put_resp_header("link", "</js/app.js>; rel=preload; as=script")
    else
      conn
    end
  end
end

defmodule ProteinAPI.KeepAliveHandler do
  @moduledoc """
  Optimized keep-alive connection handling
  """

  import Plug.Conn

  def init(opts), do: opts

  def call(conn, _opts) do
    conn
    |> configure_keep_alive()
    |> handle_connection_upgrade()
  end

  defp configure_keep_alive(conn) do
    # Set keep-alive parameters based on client capabilities
    user_agent = get_req_header(conn, "user-agent") |> List.first() || ""

    {timeout, max_requests} = if String.contains?(user_agent, ["Chrome", "Firefox", "Safari"]) do
      {120, 1000}  # Modern browsers can handle longer connections
    else
      {60, 100}    # Conservative settings for other clients
    end

    conn
    |> put_resp_header("connection", "keep-alive")
    |> put_resp_header("keep-alive", "timeout=#{timeout}, max=#{max_requests}")
  end

  defp handle_connection_upgrade(conn) do
    case get_req_header(conn, "upgrade") do
      ["h2c"] ->
        # Handle HTTP/2 upgrade
        conn
        |> put_resp_header("upgrade", "h2c")
        |> put_resp_header("connection", "upgrade")

      ["websocket"] ->
        # Handle WebSocket upgrade if needed
        conn

      _ ->
        conn
    end
  end
end

defmodule ProteinAPI.StreamingBodyReader do
  @moduledoc """
  Optimized streaming body reader for large payloads
  """

  def read_body(conn, opts) do
    length = Keyword.get(opts, :length, 8_000_000)
    read_length = Keyword.get(opts, :read_length, 1_000_000)
    read_timeout = Keyword.get(opts, :read_timeout, 15_000)

    case read_body_stream(conn, "", 0, length, read_length, read_timeout) do
      {:ok, body, conn} -> {:ok, body, conn}
      {:more, partial_body, conn} -> {:more, partial_body, conn}
      {:error, reason} -> {:error, reason}
    end
  end

  defp read_body_stream(conn, acc, acc_length, max_length, read_length, timeout) do
    if acc_length >= max_length do
      {:ok, acc, conn}
    else
      case Plug.Conn.read_body(conn,
        length: min(read_length, max_length - acc_length),
        read_timeout: timeout
      ) do
        {:ok, data, conn} ->
          {:ok, acc <> data, conn}

        {:more, data, conn} ->
          new_acc = acc <> data
          new_length = acc_length + byte_size(data)

          if new_length >= max_length do
            {:more, new_acc, conn}
          else
            read_body_stream(conn, new_acc, new_length, max_length, read_length, timeout)
          end

        {:error, reason} ->
          {:error, reason}
      end
    end
  end
end

# =======================================================================


# =======================================================================
