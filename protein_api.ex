# =======================================================================


defmodule ProteinAPI.Application do
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {Phoenix.PubSub, name: ProteinAPI.PubSub},
      ProteinAPIWeb.Endpoint,
      {Task.Supervisor, name: ProteinAPI.TaskSupervisor},
      ProteinAPI.StructureCache,
      ProteinAPI.ComputationManager,
      ProteinAPI.JuliaPort
    ]

    opts = [strategy: :one_for_one, name: ProteinAPI.Supervisor]
    Supervisor.start_link(children, opts)
  end

  @impl true
  def config_change(changed, _new, removed) do
    ProteinAPIWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end

defmodule ProteinAPI.JuliaPort do
  use GenServer
  require Logger

  @julia_script "main.jl"

  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def fold_protein(sequence, options \\ %{}) do
    GenServer.call(__MODULE__, {:fold_protein, sequence, options}, 300_000)
  end

  def validate_structure(structure_data) do
    GenServer.call(__MODULE__, {:validate_structure, structure_data}, 60_000)
  end

  def compute_confidence(structure_data) do
    GenServer.call(__MODULE__, {:compute_confidence, structure_data}, 60_000)
  end

  @impl true
  def init(_) do
    port = Port.open({:spawn, "julia #{@julia_script}"},
      [:binary, :exit_status, packet: 4])
    {:ok, %{port: port, requests: %{}}}
  end

  @impl true
  def handle_call({:fold_protein, sequence, options}, from, state) do
    request_id = :erlang.unique_integer([:positive])
    command = %{
      action: "fold_protein",
      request_id: request_id,
      sequence: sequence,
      options: options
    }

    send_command(state.port, command)
    requests = Map.put(state.requests, request_id, from)
    {:noreply, %{state | requests: requests}}
  end

  @impl true
  def handle_call({:validate_structure, structure_data}, from, state) do
    request_id = :erlang.unique_integer([:positive])
    command = %{
      action: "validate_structure",
      request_id: request_id,
      structure: structure_data
    }

    send_command(state.port, command)
    requests = Map.put(state.requests, request_id, from)
    {:noreply, %{state | requests: requests}}
  end

  @impl true
  def handle_call({:compute_confidence, structure_data}, from, state) do
    request_id = :erlang.unique_integer([:positive])
    command = %{
      action: "compute_confidence",
      request_id: request_id,
      structure: structure_data
    }

    send_command(state.port, command)
    requests = Map.put(state.requests, request_id, from)
    {:noreply, %{state | requests: requests}}
  end

  @impl true
  def handle_info({port, {:data, data}}, %{port: port} = state) do
    case Jason.decode(data) do
      {:ok, %{"request_id" => request_id, "result" => result}} ->
        case Map.pop(state.requests, request_id) do
          {nil, requests} ->
            Logger.warning("Received response for unknown request: #{request_id}")
            {:noreply, %{state | requests: requests}}
          {from, requests} ->
            GenServer.reply(from, {:ok, result})
            {:noreply, %{state | requests: requests}}
        end
      {:ok, %{"request_id" => request_id, "error" => error}} ->
        case Map.pop(state.requests, request_id) do
          {nil, requests} ->
            {:noreply, %{state | requests: requests}}
          {from, requests} ->
            GenServer.reply(from, {:error, error})
            {:noreply, %{state | requests: requests}}
        end
      {:error, _} ->
        Logger.error("Failed to decode Julia response: #{inspect(data)}")
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.error("Julia process exited with status: #{status}")
    {:stop, :julia_process_died, state}
  end

  defp send_command(port, command) do
    json = Jason.encode!(command)
    Port.command(port, json)
  end
end

defmodule ProteinAPI.StructureCache do
  use GenServer
  require Logger

  @cache_ttl :timer.hours(24)

  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def get(key) do
    GenServer.call(__MODULE__, {:get, key})
  end

  def put(key, value) do
    GenServer.cast(__MODULE__, {:put, key, value})
  end

  def delete(key) do
    GenServer.cast(__MODULE__, {:delete, key})
  end

  @impl true
  def init(_) do
    :ets.new(:structure_cache, [:set, :public, :named_table])
    schedule_cleanup()
    {:ok, %{}}
  end

  @impl true
  def handle_call({:get, key}, _from, state) do
    case :ets.lookup(:structure_cache, key) do
      [{^key, value, timestamp}] ->
        if :erlang.system_time(:millisecond) - timestamp < @cache_ttl do
          {:reply, {:ok, value}, state}
        else
          :ets.delete(:structure_cache, key)
          {:reply, :not_found, state}
        end
      [] ->
        {:reply, :not_found, state}
    end
  end

  @impl true
  def handle_cast({:put, key, value}, state) do
    timestamp = :erlang.system_time(:millisecond)
    :ets.insert(:structure_cache, {key, value, timestamp})
    {:noreply, state}
  end

  @impl true
  def handle_cast({:delete, key}, state) do
    :ets.delete(:structure_cache, key)
    {:noreply, state}
  end

  @impl true
  def handle_info(:cleanup, state) do
    current_time = :erlang.system_time(:millisecond)
    expired_keys =
      :ets.tab2list(:structure_cache)
      |> Enum.filter(fn {_, _, timestamp} -> current_time - timestamp >= @cache_ttl end)
      |> Enum.map(fn {key, _, _} -> key end)

    Enum.each(expired_keys, &:ets.delete(:structure_cache, &1))
    schedule_cleanup()
    {:noreply, state}
  end

  defp schedule_cleanup do
    Process.send_after(self(), :cleanup, @cache_ttl)
  end
end

defmodule ProteinAPI.ComputationManager do
  use GenServer
  require Logger

  @max_concurrent_jobs 4

  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def submit_job(job_data) do
    GenServer.call(__MODULE__, {:submit_job, job_data})
  end

  def get_job_status(job_id) do
    GenServer.call(__MODULE__, {:get_status, job_id})
  end

  def cancel_job(job_id) do
    GenServer.cast(__MODULE__, {:cancel_job, job_id})
  end

  @impl true
  def init(_) do
    {:ok, %{
      jobs: %{},
      queue: :queue.new(),
      running: MapSet.new()
    }}
  end

  @impl true
  def handle_call({:submit_job, job_data}, _from, state) do
    job_id = generate_job_id()
    job = %{
      id: job_id,
      data: job_data,
      status: :queued,
      created_at: DateTime.utc_now(),
      started_at: nil,
      completed_at: nil,
      result: nil,
      error: nil
    }

    jobs = Map.put(state.jobs, job_id, job)
    queue = :queue.in(job_id, state.queue)

    state = %{state | jobs: jobs, queue: queue}
    state = maybe_start_job(state)

    {:reply, {:ok, job_id}, state}
  end

  @impl true
  def handle_call({:get_status, job_id}, _from, state) do
    case Map.get(state.jobs, job_id) do
      nil -> {:reply, {:error, :not_found}, state}
      job -> {:reply, {:ok, job}, state}
    end
  end

  @impl true
  def handle_cast({:cancel_job, job_id}, state) do
    case Map.get(state.jobs, job_id) do
      nil ->
        {:noreply, state}
      %{status: :queued} ->
        jobs = Map.put(state.jobs, job_id,
          Map.merge(state.jobs[job_id], %{status: :cancelled, completed_at: DateTime.utc_now()}))
        queue = :queue.filter(fn id -> id != job_id end, state.queue)
        {:noreply, %{state | jobs: jobs, queue: queue}}
      %{status: :running} ->
        # Cancel running job (implementation depends on Julia port)
        jobs = Map.put(state.jobs, job_id,
          Map.merge(state.jobs[job_id], %{status: :cancelled, completed_at: DateTime.utc_now()}))
        running = MapSet.delete(state.running, job_id)
        state = %{state | jobs: jobs, running: running}
        state = maybe_start_job(state)
        {:noreply, state}
      _ ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:job_completed, job_id, result}, state) do
    case Map.get(state.jobs, job_id) do
      nil ->
        {:noreply, state}
      job ->
        updated_job = %{job |
          status: :completed,
          completed_at: DateTime.utc_now(),
          result: result
        }
        jobs = Map.put(state.jobs, job_id, updated_job)
        running = MapSet.delete(state.running, job_id)

        # Notify subscribers
        Phoenix.PubSub.broadcast(ProteinAPI.PubSub, "job:#{job_id}", {:job_completed, updated_job})

        state = %{state | jobs: jobs, running: running}
        state = maybe_start_job(state)
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:job_failed, job_id, error}, state) do
    case Map.get(state.jobs, job_id) do
      nil ->
        {:noreply, state}
      job ->
        updated_job = %{job |
          status: :failed,
          completed_at: DateTime.utc_now(),
          error: error
        }
        jobs = Map.put(state.jobs, job_id, updated_job)
        running = MapSet.delete(state.running, job_id)

        Phoenix.PubSub.broadcast(ProteinAPI.PubSub, "job:#{job_id}", {:job_failed, updated_job})

        state = %{state | jobs: jobs, running: running}
        state = maybe_start_job(state)
        {:noreply, state}
    end
  end

  defp maybe_start_job(state) do
    if MapSet.size(state.running) < @max_concurrent_jobs and not :queue.is_empty(state.queue) do
      {{:value, job_id}, queue} = :queue.out(state.queue)

      case Map.get(state.jobs, job_id) do
        %{status: :queued} = job ->
          updated_job = %{job | status: :running, started_at: DateTime.utc_now()}
          jobs = Map.put(state.jobs, job_id, updated_job)
          running = MapSet.put(state.running, job_id)

          # Start the actual computation
          start_computation(job_id, job.data)

          %{state | jobs: jobs, queue: queue, running: running}
        _ ->
          # Job was cancelled or already processed
          %{state | queue: queue}
      end
    else
      state
    end
  end

  defp start_computation(job_id, job_data) do
    Task.Supervisor.start_child(ProteinAPI.TaskSupervisor, fn ->
      case job_data.type do
        "fold_protein" ->
          case ProteinAPI.JuliaPort.fold_protein(job_data.sequence, job_data.options) do
            {:ok, result} -> send(self(), {:job_completed, job_id, result})
            {:error, error} -> send(self(), {:job_failed, job_id, error})
          end
        "validate_structure" ->
          case ProteinAPI.JuliaPort.validate_structure(job_data.structure) do
            {:ok, result} -> send(self(), {:job_completed, job_id, result})
            {:error, error} -> send(self(), {:job_failed, job_id, error})
          end
        _ ->
          send(self(), {:job_failed, job_id, "Unknown job type"})
      end
    end)
  end

  defp generate_job_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end
end

defmodule ProteinAPIWeb.Endpoint do
  use Phoenix.Endpoint, otp_app: :protein_api

  socket "/live", Phoenix.LiveView.Socket, websocket: [connect_info: [session: @session_options]]
  socket "/socket", ProteinAPIWeb.UserSocket, websocket: true, longpoll: false

  plug Plug.Static,
    at: "/",
    from: :protein_api,
    gzip: false,
    only: ~w(assets fonts images favicon.ico robots.txt)

  plug Plug.RequestId
  plug Plug.Telemetry, event_prefix: [:phoenix, :endpoint]

  plug Plug.Parsers,
    parsers: [:urlencoded, :multipart, :json],
    pass: ["*/*"],
    json_decoder: Phoenix.json_library()

  plug Plug.MethodOverride
  plug Plug.Head
  plug Plug.Session, @session_options
  plug ProteinAPIWeb.Router
end

defmodule ProteinAPIWeb.Router do
  use ProteinAPIWeb, :router

  pipeline :api do
    plug :accepts, ["json"]
    plug :put_resp_header, "access-control-allow-origin", "*"
    plug :put_resp_header, "access-control-allow-methods", "GET, POST, PUT, DELETE, OPTIONS"
    plug :put_resp_header, "access-control-allow-headers", "authorization, content-type"
  end

  scope "/api", ProteinAPIWeb do
    pipe_through :api

    post "/fold", ProteinController, :fold_protein
    post "/validate", ProteinController, :validate_structure
    post "/confidence", ProteinController, :compute_confidence
    get "/job/:id", JobController, :get_status
    delete "/job/:id", JobController, :cancel
    get "/structures", StructureController, :list
    get "/structures/:id", StructureController, :get
  end

  scope "/" do
    pipe_through [:api]
    get "/health", ProteinAPIWeb.HealthController, :check
  end
end

defmodule ProteinAPIWeb.ProteinController do
  use ProteinAPIWeb, :controller
  require Logger

  def fold_protein(conn, %{"sequence" => sequence} = params) do
    options = Map.get(params, "options", %{})

    # Validate sequence
    case validate_protein_sequence(sequence) do
      :ok ->
        job_data = %{
          type: "fold_protein",
          sequence: sequence,
          options: options
        }

        case ProteinAPI.ComputationManager.submit_job(job_data) do
          {:ok, job_id} ->
            conn
            |> put_status(:accepted)
            |> json(%{job_id: job_id, status: "queued"})
          {:error, reason} ->
            conn
            |> put_status(:internal_server_error)
            |> json(%{error: reason})
        end
      {:error, reason} ->
        conn
        |> put_status(:bad_request)
        |> json(%{error: reason})
    end
  end

  def validate_structure(conn, %{"structure" => structure_data}) do
    job_data = %{
      type: "validate_structure",
      structure: structure_data
    }

    case ProteinAPI.ComputationManager.submit_job(job_data) do
      {:ok, job_id} ->
        conn
        |> put_status(:accepted)
        |> json(%{job_id: job_id, status: "queued"})
      {:error, reason} ->
        conn
        |> put_status(:internal_server_error)
        |> json(%{error: reason})
    end
  end

  def compute_confidence(conn, %{"structure" => structure_data}) do
    case ProteinAPI.JuliaPort.compute_confidence(structure_data) do
      {:ok, result} ->
        json(conn, %{confidence: result})
      {:error, reason} ->
        conn
        |> put_status(:internal_server_error)
        |> json(%{error: reason})
    end
  end

  defp validate_protein_sequence(sequence) do
    valid_aa = MapSet.new(~w(A R N D C Q E G H I L K M F P S T W Y V))

    if String.length(sequence) > 0 and String.length(sequence) < 10000 do
      sequence
      |> String.upcase()
      |> String.graphemes()
      |> Enum.all?(&MapSet.member?(valid_aa, &1))
      |> case do
        true -> :ok
        false -> {:error, "Invalid amino acid sequence"}
      end
    else
      {:error, "Sequence length must be between 1 and 10000 characters"}
    end
  end
end

defmodule ProteinAPIWeb.JobController do
  use ProteinAPIWeb, :controller

  def get_status(conn, %{"id" => job_id}) do
    case ProteinAPI.ComputationManager.get_job_status(job_id) do
      {:ok, job} -> json(conn, job)
      {:error, :not_found} ->
        conn
        |> put_status(:not_found)
        |> json(%{error: "Job not found"})
    end
  end

  def cancel(conn, %{"id" => job_id}) do
    ProteinAPI.ComputationManager.cancel_job(job_id)
    json(conn, %{status: "cancelled"})
  end
end

defmodule ProteinAPIWeb.StructureController do
  use ProteinAPIWeb, :controller

  def list(conn, _params) do
    # Return list of available structures
    json(conn, %{structures: []})
  end

  def get(conn, %{"id" => structure_id}) do
    case ProteinAPI.StructureCache.get(structure_id) do
      {:ok, structure} -> json(conn, structure)
      :not_found ->
        conn
        |> put_status(:not_found)
        |> json(%{error: "Structure not found"})
    end
  end
end

defmodule ProteinAPIWeb.HealthController do
  use ProteinAPIWeb, :controller

  def check(conn, _params) do
    json(conn, %{
      status: "healthy",
      timestamp: DateTime.utc_now(),
      services: %{
        julia_port: check_julia_port(),
        cache: check_cache(),
        computation_manager: check_computation_manager()
      }
    })
  end

  defp check_julia_port do
    try do
      # Simple ping to Julia port
      case ProteinAPI.JuliaPort.compute_confidence(%{}) do
        {:ok, _} -> "healthy"
        {:error, _} -> "degraded"
      end
    rescue
      _ -> "unhealthy"
    end
  end

  defp check_cache do
    try do
      test_key = "health_check_#{:erlang.unique_integer()}"
      ProteinAPI.StructureCache.put(test_key, %{test: true})
      case ProteinAPI.StructureCache.get(test_key) do
        {:ok, _} ->
          ProteinAPI.StructureCache.delete(test_key)
          "healthy"
        :not_found -> "degraded"
      end
    rescue
      _ -> "unhealthy"
    end
  end

  defp check_computation_manager do
    try do
      case Process.whereis(ProteinAPI.ComputationManager) do
        nil -> "unhealthy"
        _pid -> "healthy"
      end
    rescue
      _ -> "unhealthy"
    end
  end
end

defmodule ProteinAPIWeb do
  def controller do
    quote do
      use Phoenix.Controller,
        formats: [:json],
        layouts: [json: ProteinAPIWeb.Layouts]

      import Plug.Conn
      import ProteinAPIWeb.Gettext

      unquote(verified_routes())
    end
  end

  def verified_routes do
    quote do
      use Phoenix.VerifiedRoutes,
        endpoint: ProteinAPIWeb.Endpoint,
        router: ProteinAPIWeb.Router,
        statics: ProteinAPIWeb.static_paths()
    end
  end

  defmacro __using__(which) when is_atom(which) do
    apply(__MODULE__, which, [])
  end
end

# =======================================================================


# =======================================================================
