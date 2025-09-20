# =======================================================================


defmodule ProteinAPI.Observability do
  @moduledoc """
  Comprehensive observability module with Telemetry, Prometheus/OTel export, and dashboard
  """

  use Application
  require Logger

  def start(_type, _args) do
    children = [
      {TelemetryRegistry, []},
      {PrometheusExporter, []},
      {OTelExporter, []},
      {DashboardServer, []}
    ]

    opts = [strategy: :one_for_one, name: ProteinAPI.Observability.Supervisor]
    Supervisor.start_link(children, opts)
  end

  def setup_telemetry do
    # Attach telemetry handlers for all components
    :telemetry.attach_many(
      "protein-api-telemetry",
      [
        [:protein_api, :request, :start],
        [:protein_api, :request, :stop],
        [:protein_api, :request, :exception],
        [:protein_api, :computation_manager, :job_queued],
        [:protein_api, :computation_manager, :job_started],
        [:protein_api, :computation_manager, :job_completed],
        [:protein_api, :computation_manager, :job_failed],
        [:protein_api, :julia_port, :command_sent],
        [:protein_api, :julia_port, :response_received],
        [:protein_api, :julia_port, :error],
        [:protein_api, :cache, :hit],
        [:protein_api, :cache, :miss],
        [:protein_api, :cache, :write],
        [:protein_api, :quantum, :job_submitted],
        [:protein_api, :quantum, :job_completed],
        [:vm, :memory],
        [:vm, :total_run_queue_lengths]
      ],
      &handle_telemetry_event/4,
      %{}
    )

    # Setup periodic VM metrics
    :telemetry_poller.start_link(
      measurements: [
        {ProteinAPI.Observability, :vm_measurements, []},
        {ProteinAPI.Observability, :application_measurements, []}
      ],
      period: :timer.seconds(5)
    )
  end

  def handle_telemetry_event([:protein_api, :request, :start], measurements, metadata, _config) do
    Prometheus.Counter.inc(:http_requests_total, [metadata.method, metadata.route])
    :prometheus_histogram.observe(:http_request_duration_seconds, [], 0)

    :opentelemetry.set_current_span(
      :otel_tracer.start_span(:default, "http_request", %{
        "http.method" => metadata.method,
        "http.route" => metadata.route,
        "http.request_id" => metadata.request_id
      })
    )
  end

  def handle_telemetry_event([:protein_api, :request, :stop], measurements, metadata, _config) do
    duration_ms = System.convert_time_unit(measurements.duration, :native, :millisecond)

    Prometheus.Histogram.observe(:http_request_duration_seconds, [metadata.method, metadata.route, metadata.status], duration_ms / 1000)
    Prometheus.Counter.inc(:http_requests_total, [metadata.method, metadata.route, metadata.status])

    :otel_span.set_attributes([
      {"http.status_code", metadata.status},
      {"http.response_size", measurements.response_size || 0}
    ])
    :otel_span.end_span()
  end

  def handle_telemetry_event([:protein_api, :computation_manager, :job_queued], measurements, metadata, _config) do
    Prometheus.Counter.inc(:jobs_total, ["queued", metadata.job_type])
    Prometheus.Gauge.inc(:queue_depth, [metadata.job_type])

    :otel_tracer.start_span(:default, "job_queued", %{
      "job.id" => metadata.job_id,
      "job.type" => metadata.job_type,
      "job.queue_depth" => measurements.queue_depth
    })
  end

  def handle_telemetry_event([:protein_api, :computation_manager, :job_started], measurements, metadata, _config) do
    queue_time_ms = System.convert_time_unit(measurements.queue_time, :native, :millisecond)

    Prometheus.Counter.inc(:jobs_total, ["started", metadata.job_type])
    Prometheus.Gauge.dec(:queue_depth, [metadata.job_type])
    Prometheus.Histogram.observe(:job_queue_duration_seconds, [metadata.job_type], queue_time_ms / 1000)

    :otel_span.set_attributes([
      {"job.queue_time_ms", queue_time_ms},
      {"job.priority", metadata.priority}
    ])
  end

  def handle_telemetry_event([:protein_api, :computation_manager, :job_completed], measurements, metadata, _config) do
    execution_time_ms = System.convert_time_unit(measurements.execution_time, :native, :millisecond)

    Prometheus.Counter.inc(:jobs_total, ["completed", metadata.job_type])
    Prometheus.Histogram.observe(:job_execution_duration_seconds, [metadata.job_type], execution_time_ms / 1000)

    :otel_span.set_attributes([
      {"job.execution_time_ms", execution_time_ms},
      {"job.result_size", measurements.result_size || 0}
    ])
    :otel_span.end_span()
  end

  def handle_telemetry_event([:protein_api, :computation_manager, :job_failed], measurements, metadata, _config) do
    Prometheus.Counter.inc(:jobs_total, ["failed", metadata.job_type])
    Prometheus.Counter.inc(:job_errors_total, [metadata.job_type, metadata.error_type])

    :otel_span.set_attributes([
      {"job.error", metadata.error},
      {"job.error_type", metadata.error_type}
    ])
    :otel_span.record_exception(metadata.error)
    :otel_span.set_status(:error, metadata.error)
    :otel_span.end_span()
  end

  def handle_telemetry_event([:protein_api, :julia_port, :command_sent], measurements, metadata, _config) do
    Prometheus.Counter.inc(:julia_commands_total, [metadata.command_type])

    :otel_tracer.start_span(:default, "julia_command", %{
      "julia.command" => metadata.command_type,
      "julia.request_id" => metadata.request_id,
      "julia.payload_size" => measurements.payload_size
    })
  end

  def handle_telemetry_event([:protein_api, :julia_port, :response_received], measurements, metadata, _config) do
    rtt_ms = System.convert_time_unit(measurements.duration, :native, :millisecond)

    Prometheus.Histogram.observe(:julia_rtt_seconds, [metadata.command_type], rtt_ms / 1000)
    Prometheus.Counter.inc(:julia_responses_total, [metadata.command_type, "success"])

    :otel_span.set_attributes([
      {"julia.rtt_ms", rtt_ms},
      {"julia.response_size", measurements.response_size}
    ])
    :otel_span.end_span()
  end

  def handle_telemetry_event([:protein_api, :julia_port, :error], measurements, metadata, _config) do
    Prometheus.Counter.inc(:julia_responses_total, [metadata.command_type, "error"])
    Prometheus.Counter.inc(:julia_errors_total, [metadata.command_type, metadata.error_type])

    :otel_span.set_attributes([
      {"julia.error", metadata.error},
      {"julia.error_type", metadata.error_type}
    ])
    :otel_span.record_exception(metadata.error)
    :otel_span.set_status(:error, metadata.error)
    :otel_span.end_span()
  end

  def handle_telemetry_event([:protein_api, :cache, :hit], measurements, metadata, _config) do
    Prometheus.Counter.inc(:cache_operations_total, [metadata.cache_type, "hit"])

    :otel_span.add_event("cache_hit", %{
      "cache.key" => metadata.key,
      "cache.type" => metadata.cache_type
    })
  end

  def handle_telemetry_event([:protein_api, :cache, :miss], measurements, metadata, _config) do
    Prometheus.Counter.inc(:cache_operations_total, [metadata.cache_type, "miss"])

    :otel_span.add_event("cache_miss", %{
      "cache.key" => metadata.key,
      "cache.type" => metadata.cache_type
    })
  end

  def handle_telemetry_event([:protein_api, :cache, :write], measurements, metadata, _config) do
    Prometheus.Counter.inc(:cache_operations_total, [metadata.cache_type, "write"])
    Prometheus.Histogram.observe(:cache_write_duration_seconds, [metadata.cache_type], measurements.duration / 1000)

    :otel_span.add_event("cache_write", %{
      "cache.key" => metadata.key,
      "cache.type" => metadata.cache_type,
      "cache.size" => measurements.size
    })
  end

  def handle_telemetry_event([:protein_api, :quantum, :job_submitted], measurements, metadata, _config) do
    Prometheus.Counter.inc(:quantum_jobs_total, [metadata.backend, "submitted"])

    :otel_tracer.start_span(:default, "quantum_job", %{
      "quantum.job_id" => metadata.job_id,
      "quantum.backend" => metadata.backend,
      "quantum.shots" => metadata.shots,
      "quantum.circuit_depth" => measurements.circuit_depth
    })
  end

  def handle_telemetry_event([:protein_api, :quantum, :job_completed], measurements, metadata, _config) do
    execution_time_ms = System.convert_time_unit(measurements.duration, :native, :millisecond)

    Prometheus.Counter.inc(:quantum_jobs_total, [metadata.backend, metadata.status])
    Prometheus.Histogram.observe(:quantum_job_duration_seconds, [metadata.backend], execution_time_ms / 1000)

    :otel_span.set_attributes([
      {"quantum.execution_time_ms", execution_time_ms},
      {"quantum.success_probability", measurements.success_probability}
    ])
    :otel_span.end_span()
  end

  def handle_telemetry_event(_event, _measurements, _metadata, _config) do
    # Ignore unknown events
    :ok
  end

  def vm_measurements do
    %{
      memory_total: :erlang.memory(:total),
      memory_processes: :erlang.memory(:processes),
      memory_atom: :erlang.memory(:atom),
      memory_binary: :erlang.memory(:binary),
      memory_ets: :erlang.memory(:ets),
      process_count: :erlang.system_info(:process_count),
      run_queue: :erlang.statistics(:run_queue),
      io_input: element(1, :erlang.statistics(:io)),
      io_output: element(2, :erlang.statistics(:io))
    }
  end

  def application_measurements do
    %{
      cache_size: ProteinAPI.StructureCache.size(),
      active_jobs: ProteinAPI.ComputationManager.active_job_count(),
      queue_depth: ProteinAPI.ComputationManager.queue_depth(),
      julia_port_status: ProteinAPI.JuliaPort.status()
    }
  end
end

defmodule ProteinAPI.Observability.PrometheusExporter do
  use GenServer
  require Logger

  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def init(_) do
    setup_prometheus_metrics()
    {:ok, %{}}
  end

  defp setup_prometheus_metrics do
    # HTTP metrics
    Prometheus.Counter.declare([
      name: :http_requests_total,
      help: "Total HTTP requests",
      labels: [:method, :route, :status]
    ])

    Prometheus.Histogram.declare([
      name: :http_request_duration_seconds,
      help: "HTTP request duration",
      labels: [:method, :route, :status],
      buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
    ])

    # Job metrics
    Prometheus.Counter.declare([
      name: :jobs_total,
      help: "Total jobs processed",
      labels: [:status, :type]
    ])

    Prometheus.Gauge.declare([
      name: :queue_depth,
      help: "Current queue depth",
      labels: [:type]
    ])

    Prometheus.Histogram.declare([
      name: :job_queue_duration_seconds,
      help: "Time spent in queue",
      labels: [:type],
      buckets: [0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300]
    ])

    Prometheus.Histogram.declare([
      name: :job_execution_duration_seconds,
      help: "Job execution time",
      labels: [:type],
      buckets: [0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300]
    ])

    Prometheus.Counter.declare([
      name: :job_errors_total,
      help: "Total job errors",
      labels: [:type, :error_type]
    ])

    # Julia Port metrics
    Prometheus.Counter.declare([
      name: :julia_commands_total,
      help: "Total Julia commands sent",
      labels: [:command_type]
    ])

    Prometheus.Counter.declare([
      name: :julia_responses_total,
      help: "Total Julia responses",
      labels: [:command_type, :status]
    ])

    Prometheus.Histogram.declare([
      name: :julia_rtt_seconds,
      help: "Julia command round-trip time",
      labels: [:command_type],
      buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]
    ])

    Prometheus.Counter.declare([
      name: :julia_errors_total,
      help: "Total Julia errors",
      labels: [:command_type, :error_type]
    ])

    # Cache metrics
    Prometheus.Counter.declare([
      name: :cache_operations_total,
      help: "Total cache operations",
      labels: [:type, :operation]
    ])

    Prometheus.Histogram.declare([
      name: :cache_write_duration_seconds,
      help: "Cache write duration",
      labels: [:type],
      buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    ])

    # Quantum metrics
    Prometheus.Counter.declare([
      name: :quantum_jobs_total,
      help: "Total quantum jobs",
      labels: [:backend, :status]
    ])

    Prometheus.Histogram.declare([
      name: :quantum_job_duration_seconds,
      help: "Quantum job execution time",
      labels: [:backend],
      buckets: [1, 5, 10, 30, 60, 120, 300, 600, 1200, 3600]
    ])

    # VM metrics
    Prometheus.Gauge.declare([
      name: :erlang_vm_memory_bytes,
      help: "Erlang VM memory usage",
      labels: [:type]
    ])

    Prometheus.Gauge.declare([
      name: :erlang_vm_process_count,
      help: "Number of Erlang processes"
    ])

    Logger.info("Prometheus metrics initialized")
  end
end

defmodule ProteinAPI.Observability.OTelExporter do
  use GenServer
  require Logger

  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def init(_) do
    setup_opentelemetry()
    {:ok, %{}}
  end

  defp setup_opentelemetry do
    # Configure OpenTelemetry
    :opentelemetry.set_default_tracer({:otel_tracer_default, :protein_api})

    # Setup resource attributes
    :otel_resource.set_attributes([
      {"service.name", "protein-api"},
      {"service.version", Application.spec(:protein_api, :vsn)},
      {"deployment.environment", Application.get_env(:protein_api, :environment, "development")}
    ])

    # Configure exporters
    jaeger_endpoint = Application.get_env(:protein_api, :jaeger_endpoint, "http://localhost:14268/api/traces")

    :opentelemetry_exporter.setup_jaeger(
      endpoint: jaeger_endpoint,
      service_name: "protein-api"
    )

    Logger.info("OpenTelemetry configured with Jaeger export to #{jaeger_endpoint}")
  end
end

defmodule ProteinAPI.Observability.DashboardServer do
  use GenServer
  require Logger

  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def init(_) do
    port = Application.get_env(:protein_api, :dashboard_port, 9090)

    {:ok, _} = :cowboy.start_clear(
      :dashboard_http,
      [{:port, port}],
      %{env: %{dispatch: compile_routes()}}
    )

    Logger.info("Dashboard server started on port #{port}")
    {:ok, %{port: port}}
  end

  defp compile_routes do
    :cowboy_router.compile([
      {'_', [
        {"/metrics", ProteinAPI.Observability.MetricsHandler, []},
        {"/health", ProteinAPI.Observability.HealthHandler, []},
        {"/dashboard", ProteinAPI.Observability.DashboardHandler, []},
        {"/api/metrics", ProteinAPI.Observability.MetricsAPIHandler, []}
      ]}
    ])
  end
end

defmodule ProteinAPI.Observability.MetricsHandler do
  def init(req, state) do
    metrics = Prometheus.Format.Text.format()

    req2 = :cowboy_req.reply(200, %{
      "content-type" => "text/plain; version=0.0.4; charset=utf-8"
    }, metrics, req)

    {:ok, req2, state}
  end
end

defmodule ProteinAPI.Observability.HealthHandler do
  def init(req, state) do
    health_status = %{
      status: "healthy",
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      services: %{
        julia_port: ProteinAPI.JuliaPort.health_check(),
        cache: ProteinAPI.StructureCache.health_check(),
        computation_manager: ProteinAPI.ComputationManager.health_check()
      }
    }

    req2 = :cowboy_req.reply(200, %{
      "content-type" => "application/json"
    }, Jason.encode!(health_status), req)

    {:ok, req2, state}
  end
end

defmodule ProteinAPI.Observability.DashboardHandler do
  def init(req, state) do
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Protein API Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .metric-box { border: 1px solid #ddd; padding: 10px; margin: 10px; display: inline-block; }
            .chart { width: 45%; height: 400px; display: inline-block; margin: 2.5%; }
        </style>
    </head>
    <body>
        <h1>Protein API Dashboard</h1>

        <div class="metric-box">
            <h3>Queue Depth</h3>
            <div id="queue-depth">Loading...</div>
        </div>

        <div class="metric-box">
            <h3>Active Jobs</h3>
            <div id="active-jobs">Loading...</div>
        </div>

        <div class="metric-box">
            <h3>Julia Port Status</h3>
            <div id="julia-status">Loading...</div>
        </div>

        <div class="chart" id="request-rate"></div>
        <div class="chart" id="response-times"></div>
        <div class="chart" id="error-rates"></div>
        <div class="chart" id="memory-usage"></div>

        <script>
            function updateMetrics() {
                fetch('/api/metrics')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('queue-depth').textContent = data.queue_depth;
                        document.getElementById('active-jobs').textContent = data.active_jobs;
                        document.getElementById('julia-status').textContent = data.julia_status;

                        updateCharts(data);
                    });
            }

            function updateCharts(data) {
                // Update request rate chart
                Plotly.newPlot('request-rate', [{
                    y: data.request_rates,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Requests/sec'
                }], {title: 'Request Rate'});

                // Update response times chart
                Plotly.newPlot('response-times', [{
                    y: data.response_times,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Response Time (ms)'
                }], {title: 'Response Times'});

                // Update error rates chart
                Plotly.newPlot('error-rates', [{
                    y: data.error_rates,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Errors/sec'
                }], {title: 'Error Rate'});

                // Update memory usage chart
                Plotly.newPlot('memory-usage', [{
                    y: data.memory_usage,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Memory (MB)'
                }], {title: 'Memory Usage'});
            }

            updateMetrics();
            setInterval(updateMetrics, 5000);
        </script>
    </body>
    </html>
    """

    req2 = :cowboy_req.reply(200, %{
      "content-type" => "text/html"
    }, dashboard_html, req)

    {:ok, req2, state}
  end
end

defmodule ProteinAPI.Observability.MetricsAPIHandler do
  def init(req, state) do
    metrics = %{
      queue_depth: ProteinAPI.ComputationManager.queue_depth(),
      active_jobs: ProteinAPI.ComputationManager.active_job_count(),
      julia_status: ProteinAPI.JuliaPort.status(),
      request_rates: get_request_rates(),
      response_times: get_response_times(),
      error_rates: get_error_rates(),
      memory_usage: get_memory_usage()
    }

    req2 = :cowboy_req.reply(200, %{
      "content-type" => "application/json"
    }, Jason.encode!(metrics), req)

    {:ok, req2, state}
  end

  defp get_request_rates do
    # Implementation for historical request rate data
    []
  end

  defp get_response_times do
    # Implementation for historical response time data
    []
  end

  defp get_error_rates do
    # Implementation for historical error rate data
    []
  end

  defp get_memory_usage do
    # Implementation for historical memory usage data
    []
  end
end

# =======================================================================


# =======================================================================
