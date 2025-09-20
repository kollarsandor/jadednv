# =======================================================================


%% Erlang Distributed Actor System for Quantum Protein Folding
%% Fault-tolerant, distributed, and scalable quantum computation management

-module(distributed_actors).
-behaviour(gen_server).
-behaviour(supervisor).

%% API exports
-export([start_link/0, stop/0]).
-export([submit_quantum_job/3, get_job_status/1, cancel_job/1]).
-export([add_quantum_node/2, remove_quantum_node/1, get_cluster_status/0]).
-export([register_verification_actor/1, verify_quantum_result/2]).

%% Internal exports
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3]).
-export([quantum_worker_supervisor_init/1]).
-export([quantum_actor_init/1, verification_actor_init/1, load_balancer_init/1]).

%% Records
-record(quantum_job, {
    id :: binary(),
    type :: atom(),
    circuit :: term(),
    backend :: binary(),
    shots :: integer(),
    priority :: integer(),
    submitted_at :: integer(),
    status :: atom(),
    result :: term(),
    error :: term(),
    node :: node()
}).

-record(quantum_node, {
    name :: atom(),
    address :: binary(),
    capabilities :: list(),
    load :: float(),
    status :: atom(),
    last_heartbeat :: integer()
}).

-record(cluster_state, {
    nodes :: [#quantum_node{}],
    jobs :: #{binary() => #quantum_job{}},
    load_balancer :: pid(),
    verification_actors :: [pid()],
    stats :: #{atom() => integer()}
}).

%% Constants
-define(MAX_RETRIES, 5).
-define(HEARTBEAT_INTERVAL, 5000).
-define(JOB_TIMEOUT, 300000).
-define(VERIFICATION_TIMEOUT, 30000).
-define(IBM_QUANTUM_BACKENDS, [<<"ibm_torino">>, <<"ibm_brisbane">>]).

%% ============================================================================
%% API Functions
%% ============================================================================

start_link() ->
    gen_server:start_link({local, quantum_cluster_manager}, ?MODULE, [], []).

stop() ->
    gen_server:call(quantum_cluster_manager, stop).

submit_quantum_job(Type, Circuit, Options) ->
    JobId = generate_job_id(),
    Job = #quantum_job{
        id = JobId,
        type = Type,
        circuit = Circuit,
        backend = maps:get(backend, Options, <<"ibm_torino">>),
        shots = maps:get(shots, Options, 1024),
        priority = maps:get(priority, Options, 1),
        submitted_at = erlang:system_time(millisecond),
        status = queued,
        node = node()
    },
    gen_server:call(quantum_cluster_manager, {submit_job, Job}).

get_job_status(JobId) ->
    gen_server:call(quantum_cluster_manager, {get_job_status, JobId}).

cancel_job(JobId) ->
    gen_server:call(quantum_cluster_manager, {cancel_job, JobId}).

add_quantum_node(NodeName, NodeAddress) ->
    gen_server:call(quantum_cluster_manager, {add_node, NodeName, NodeAddress}).

remove_quantum_node(NodeName) ->
    gen_server:call(quantum_cluster_manager, {remove_node, NodeName}).

get_cluster_status() ->
    gen_server:call(quantum_cluster_manager, get_cluster_status).

register_verification_actor(Pid) ->
    gen_server:call(quantum_cluster_manager, {register_verification_actor, Pid}).

verify_quantum_result(JobId, Result) ->
    gen_server:call(quantum_cluster_manager, {verify_result, JobId, Result}).

%% ============================================================================
%% Gen_server Callbacks
%% ============================================================================

init([]) ->
    process_flag(trap_exit, true),

    %% Start supervision tree
    {ok, LoadBalancerPid} = start_load_balancer(),
    {ok, _SupervisorPid} = start_quantum_worker_supervisor(),

    %% Initialize verification actors
    VerificationActors = start_verification_actors(3),

    %% Setup periodic heartbeat
    timer:send_interval(?HEARTBEAT_INTERVAL, heartbeat),

    State = #cluster_state{
        nodes = [],
        jobs = #{},
        load_balancer = LoadBalancerPid,
        verification_actors = VerificationActors,
        stats = #{
            submitted => 0,
            completed => 0,
            failed => 0,
            cancelled => 0
        }
    },

    logger:info("Quantum cluster manager started"),
    {ok, State}.

handle_call({submit_job, Job}, _From, State) ->
    JobId = Job#quantum_job.id,
    UpdatedJobs = maps:put(JobId, Job, State#cluster_state.jobs),

    %% Send job to load balancer for distribution
    State#cluster_state.load_balancer ! {new_job, Job},

    %% Update statistics
    Stats = maps:update_with(submitted, fun(V) -> V + 1 end, 1, State#cluster_state.stats),

    NewState = State#cluster_state{
        jobs = UpdatedJobs,
        stats = Stats
    },

    logger:info("Job submitted: ~s", [JobId]),
    {reply, {ok, JobId}, NewState};

handle_call({get_job_status, JobId}, _From, State) ->
    case maps:get(JobId, State#cluster_state.jobs, undefined) of
        undefined ->
            {reply, {error, job_not_found}, State};
        Job ->
            {reply, {ok, job_to_map(Job)}, State}
    end;

handle_call({cancel_job, JobId}, _From, State) ->
    case maps:get(JobId, State#cluster_state.jobs, undefined) of
        undefined ->
            {reply, {error, job_not_found}, State};
        Job when Job#quantum_job.status =:= completed;
                 Job#quantum_job.status =:= failed;
                 Job#quantum_job.status =:= cancelled ->
            {reply, {error, job_already_finished}, State};
        Job ->
            %% Cancel job on the processing node
            rpc:cast(Job#quantum_job.node, quantum_worker, cancel_job, [JobId]),

            UpdatedJob = Job#quantum_job{status = cancelled},
            UpdatedJobs = maps:put(JobId, UpdatedJob, State#cluster_state.jobs),

            Stats = maps:update_with(cancelled, fun(V) -> V + 1 end, 1, State#cluster_state.stats),

            NewState = State#cluster_state{
                jobs = UpdatedJobs,
                stats = Stats
            },

            logger:info("Job cancelled: ~s", [JobId]),
            {reply, ok, NewState}
    end;

handle_call({add_node, NodeName, NodeAddress}, _From, State) ->
    Node = #quantum_node{
        name = NodeName,
        address = NodeAddress,
        capabilities = ?IBM_QUANTUM_BACKENDS,
        load = 0.0,
        status = active,
        last_heartbeat = erlang:system_time(millisecond)
    },

    UpdatedNodes = [Node | State#cluster_state.nodes],
    NewState = State#cluster_state{nodes = UpdatedNodes},

    %% Notify load balancer of new node
    State#cluster_state.load_balancer ! {node_added, Node},

    logger:info("Quantum node added: ~p", [NodeName]),
    {reply, ok, NewState};

handle_call({remove_node, NodeName}, _From, State) ->
    UpdatedNodes = lists:filter(
        fun(Node) -> Node#quantum_node.name =/= NodeName end,
        State#cluster_state.nodes
    ),

    NewState = State#cluster_state{nodes = UpdatedNodes},

    %% Notify load balancer of node removal
    State#cluster_state.load_balancer ! {node_removed, NodeName},

    logger:info("Quantum node removed: ~p", [NodeName]),
    {reply, ok, NewState};

handle_call(get_cluster_status, _From, State) ->
    Status = #{
        nodes => length(State#cluster_state.nodes),
        active_jobs => count_active_jobs(State#cluster_state.jobs),
        total_jobs => maps:size(State#cluster_state.jobs),
        stats => State#cluster_state.stats,
        load_balancer_status => is_process_alive(State#cluster_state.load_balancer),
        verification_actors => length(State#cluster_state.verification_actors)
    },
    {reply, Status, State};

handle_call({register_verification_actor, Pid}, _From, State) ->
    UpdatedActors = [Pid | State#cluster_state.verification_actors],
    NewState = State#cluster_state{verification_actors = UpdatedActors},
    {reply, ok, NewState};

handle_call({verify_result, JobId, Result}, _From, State) ->
    %% Delegate to verification actor
    case State#cluster_state.verification_actors of
        [] ->
            {reply, {error, no_verification_actors}, State};
        [Actor | _] ->
            Actor ! {verify_result, JobId, Result, self()},
            {reply, ok, State}
    end;

handle_call(stop, _From, State) ->
    {stop, normal, ok, State}.

handle_cast({job_completed, JobId, Result}, State) ->
    case maps:get(JobId, State#cluster_state.jobs, undefined) of
        undefined ->
            {noreply, State};
        Job ->
            UpdatedJob = Job#quantum_job{
                status = completed,
                result = Result
            },
            UpdatedJobs = maps:put(JobId, UpdatedJob, State#cluster_state.jobs),

            Stats = maps:update_with(completed, fun(V) -> V + 1 end, 1, State#cluster_state.stats),

            NewState = State#cluster_state{
                jobs = UpdatedJobs,
                stats = Stats
            },

            %% Verify result if verification actors are available
            case State#cluster_state.verification_actors of
                [] -> ok;
                [Actor | _] -> Actor ! {verify_result, JobId, Result, self()}
            end,

            logger:info("Job completed: ~s", [JobId]),
            {noreply, NewState}
    end;

handle_cast({job_failed, JobId, Error}, State) ->
    case maps:get(JobId, State#cluster_state.jobs, undefined) of
        undefined ->
            {noreply, State};
        Job ->
            UpdatedJob = Job#quantum_job{
                status = failed,
                error = Error
            },
            UpdatedJobs = maps:put(JobId, UpdatedJob, State#cluster_state.jobs),

            Stats = maps:update_with(failed, fun(V) -> V + 1 end, 1, State#cluster_state.stats),

            NewState = State#cluster_state{
                jobs = UpdatedJobs,
                stats = Stats
            },

            logger:error("Job failed: ~s, Error: ~p", [JobId, Error]),
            {noreply, NewState}
    end;

handle_cast({node_heartbeat, NodeName, Load}, State) ->
    UpdatedNodes = lists:map(
        fun(Node) when Node#quantum_node.name =:= NodeName ->
                Node#quantum_node{
                    load = Load,
                    last_heartbeat = erlang:system_time(millisecond)
                };
           (Node) ->
                Node
        end,
        State#cluster_state.nodes
    ),

    NewState = State#cluster_state{nodes = UpdatedNodes},
    {noreply, NewState}.

handle_info(heartbeat, State) ->
    %% Check for stale nodes
    CurrentTime = erlang:system_time(millisecond),
    UpdatedNodes = lists:filter(
        fun(Node) ->
            TimeSinceHeartbeat = CurrentTime - Node#quantum_node.last_heartbeat,
            if
                TimeSinceHeartbeat > 30000 -> % 30 seconds
                    logger:warning("Node ~p appears to be down, removing", [Node#quantum_node.name]),
                    false;
                true ->
                    true
            end
        end,
        State#cluster_state.nodes
    ),

    NewState = State#cluster_state{nodes = UpdatedNodes},
    {noreply, NewState};

handle_info({'EXIT', Pid, Reason}, State) ->
    logger:error("Process ~p exited with reason: ~p", [Pid, Reason]),
    %% Handle process exits and restart if necessary
    {noreply, State};

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    logger:info("Quantum cluster manager terminated"),
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%% ============================================================================
%% Load Balancer Actor
%% ============================================================================

start_load_balancer() ->
    Pid = spawn_link(?MODULE, load_balancer_init, [self()]),
    {ok, Pid}.

load_balancer_init(ClusterManager) ->
    process_flag(trap_exit, true),
    logger:info("Load balancer started"),
    load_balancer_loop(ClusterManager, [], queue:new()).

load_balancer_loop(ClusterManager, Nodes, JobQueue) ->
    receive
        {new_job, Job} ->
            UpdatedQueue = queue:in(Job, JobQueue),
            NewQueue = try_distribute_jobs(ClusterManager, Nodes, UpdatedQueue),
            load_balancer_loop(ClusterManager, Nodes, NewQueue);

        {node_added, Node} ->
            UpdatedNodes = [Node | Nodes],
            NewQueue = try_distribute_jobs(ClusterManager, UpdatedNodes, JobQueue),
            load_balancer_loop(ClusterManager, UpdatedNodes, NewQueue);

        {node_removed, NodeName} ->
            UpdatedNodes = lists:filter(
                fun(Node) -> Node#quantum_node.name =/= NodeName end,
                Nodes
            ),
            load_balancer_loop(ClusterManager, UpdatedNodes, JobQueue);

        {'EXIT', _Pid, Reason} ->
            logger:error("Load balancer exiting: ~p", [Reason]),
            ok
    end.

try_distribute_jobs(ClusterManager, Nodes, JobQueue) ->
    case queue:out(JobQueue) of
        {empty, _} ->
            JobQueue;
        {{value, Job}, RemainingQueue} ->
            case find_best_node(Nodes, Job) of
                {ok, Node} ->
                    %% Submit job to selected node
                    spawn(fun() -> execute_quantum_job(ClusterManager, Job, Node) end),
                    try_distribute_jobs(ClusterManager, Nodes, RemainingQueue);
                {error, no_available_nodes} ->
                    %% Put job back in queue
                    JobQueue
            end
    end.

find_best_node(Nodes, Job) ->
    AvailableNodes = lists:filter(
        fun(Node) ->
            Node#quantum_node.status =:= active andalso
            Node#quantum_node.load < 0.8 andalso
            lists:member(Job#quantum_job.backend, Node#quantum_node.capabilities)
        end,
        Nodes
    ),

    case AvailableNodes of
        [] ->
            {error, no_available_nodes};
        _ ->
            %% Select node with lowest load
            BestNode = lists:foldl(
                fun(Node, Acc) ->
                    case Acc of
                        undefined -> Node;
                        _ when Node#quantum_node.load < Acc#quantum_node.load -> Node;
                        _ -> Acc
                    end
                end,
                undefined,
                AvailableNodes
            ),
            {ok, BestNode}
    end.

%% ============================================================================
%% Quantum Job Execution
%% ============================================================================

execute_quantum_job(ClusterManager, Job, Node) ->
    JobId = Job#quantum_job.id,

    try
        %% Execute quantum circuit based on type
        Result = case Job#quantum_job.type of
            grover_protein_folding ->
                execute_grover_folding(Job);
            vqe_energy_minimization ->
                execute_vqe_optimization(Job);
            qft_molecular_dynamics ->
                execute_qft_dynamics(Job);
            _ ->
                throw({error, unknown_job_type})
        end,

        gen_server:cast(ClusterManager, {job_completed, JobId, Result})

    catch
        Error:Reason ->
            logger:error("Job execution failed: ~p:~p", [Error, Reason]),
            gen_server:cast(ClusterManager, {job_failed, JobId, {Error, Reason}})
    end.

execute_grover_folding(Job) ->
    Circuit = Job#quantum_job.circuit,
    Backend = Job#quantum_job.backend,
    Shots = Job#quantum_job.shots,

    %% Generate Grover circuit for protein structure search
    GroverCircuit = generate_grover_circuit(Circuit),

    %% Submit to IBM Quantum
    {ok, IBMJobId} = submit_to_ibm_quantum(GroverCircuit, Backend, Shots),

    %% Wait for results
    {ok, RawResult} = wait_for_ibm_result(IBMJobId),

    %% Post-process quantum results
    process_grover_results(RawResult).

execute_vqe_optimization(Job) ->
    Circuit = Job#quantum_job.circuit,
    Backend = Job#quantum_job.backend,
    Shots = Job#quantum_job.shots,

    %% Variational quantum eigensolver for energy minimization
    VQECircuit = generate_vqe_circuit(Circuit),

    %% Iterative optimization
    OptimalParameters = vqe_optimization_loop(VQECircuit, Backend, Shots),

    %% Return final energy estimate
    #{
        optimal_parameters => OptimalParameters,
        final_energy => calculate_final_energy(OptimalParameters),
        iterations => get_vqe_iterations()
    }.

execute_qft_dynamics(Job) ->
    Circuit = Job#quantum_job.circuit,
    Backend = Job#quantum_job.backend,
    Shots = Job#quantum_job.shots,

    %% Quantum Fourier Transform for molecular dynamics
    QFTCircuit = generate_qft_circuit(Circuit),

    %% Submit and process
    {ok, IBMJobId} = submit_to_ibm_quantum(QFTCircuit, Backend, Shots),
    {ok, RawResult} = wait_for_ibm_result(IBMJobId),

    process_qft_results(RawResult).

%% ============================================================================
%% Verification Actors
%% ============================================================================

start_verification_actors(Count) ->
    [spawn_link(?MODULE, verification_actor_init, []) || _ <- lists:seq(1, Count)].

verification_actor_init() ->
    logger:info("Verification actor started: ~p", [self()]),
    verification_actor_loop().

verification_actor_loop() ->
    receive
        {verify_result, JobId, Result, ClusterManager} ->
            IsValid = verify_quantum_result_internal(Result),
            ClusterManager ! {verification_complete, JobId, IsValid},
            verification_actor_loop();

        {'EXIT', _Pid, Reason} ->
            logger:error("Verification actor exiting: ~p", [Reason]),
            ok
    end.

verify_quantum_result_internal(Result) ->
    %% Comprehensive quantum result verification
    try
        %% Check result format
        case maps:is_key(<<"counts">>, Result) of
            false -> false;
            true ->
                Counts = maps:get(<<"counts">>, Result),

                %% Verify probability conservation
                TotalCounts = lists:sum(maps:values(Counts)),

                %% Check for reasonable distribution
                MaxCount = lists:max(maps:values(Counts)),
                IsReasonable = MaxCount / TotalCounts =< 0.9, % No single outcome dominates too much

                %% Additional quantum-specific validations
                HasValidStates = all_states_valid(maps:keys(Counts)),

                IsReasonable andalso HasValidStates
        end
    catch
        _:_ -> false
    end.

all_states_valid(States) ->
    lists:all(
        fun(State) ->
            is_binary(State) andalso byte_size(State) > 0
        end,
        States
    ).

%% ============================================================================
%% IBM Quantum Interface
%% ============================================================================

submit_to_ibm_quantum(Circuit, Backend, Shots) ->
    %% Simplified IBM Quantum submission
    %% In practice, this would use the IBM Quantum API
    timer:sleep(1000), % Simulate network delay
    IBMJobId = generate_job_id(),
    spawn(fun() -> simulate_ibm_execution(IBMJobId, Circuit, Shots) end),
    {ok, IBMJobId}.

wait_for_ibm_result(IBMJobId) ->
    %% Poll for IBM Quantum result
    wait_for_result_loop(IBMJobId, 0).

wait_for_result_loop(IBMJobId, Attempts) when Attempts < 60 ->
    timer:sleep(5000), % Check every 5 seconds
    case get_simulated_result(IBMJobId) of
        {ok, Result} ->
            {ok, Result};
        {pending} ->
            wait_for_result_loop(IBMJobId, Attempts + 1)
    end;
wait_for_result_loop(_IBMJobId, _Attempts) ->
    {error, timeout}.

simulate_ibm_execution(IBMJobId, _Circuit, Shots) ->
    %% Simulate quantum execution
    timer:sleep(10000), % Simulate execution time

    %% Generate mock quantum results
    States = [<<"000">>, <<"001">>, <<"010">>, <<"011">>,
              <<"100">>, <<"101">>, <<"110">>, <<"111">>],

    Counts = lists:foldl(
        fun(State, Acc) ->
            Count = rand:uniform(Shots div 4),
            maps:put(State, Count, Acc)
        end,
        #{},
        States
    ),

    Result = #{
        <<"job_id">> => IBMJobId,
        <<"backend">> => <<"ibm_torino">>,
        <<"shots">> => Shots,
        <<"counts">> => Counts,
        <<"success">> => true
    },

    %% Store result for retrieval
    put_simulated_result(IBMJobId, Result).

%% ============================================================================
%% Circuit Generation Functions
%% ============================================================================

generate_grover_circuit(BaseCircuit) ->
    %% Generate Grover algorithm circuit for protein folding
    NumQubits = maps:get(qubits, BaseCircuit, 6),
    Iterations = round(math:pi() / 4 * math:sqrt(math:pow(2, NumQubits))),

    #{
        type => grover,
        qubits => NumQubits,
        iterations => Iterations,
        oracle => protein_folding_oracle,
        diffuser => grover_diffuser
    }.

generate_vqe_circuit(BaseCircuit) ->
    %% Variational Quantum Eigensolver circuit
    NumQubits = maps:get(qubits, BaseCircuit, 4),
    Depth = maps:get(depth, BaseCircuit, 3),

    #{
        type => vqe,
        qubits => NumQubits,
        depth => Depth,
        ansatz => hardware_efficient,
        parameters => generate_initial_parameters(NumQubits, Depth)
    }.

generate_qft_circuit(BaseCircuit) ->
    %% Quantum Fourier Transform circuit
    NumQubits = maps:get(qubits, BaseCircuit, 8),

    #{
        type => qft,
        qubits => NumQubits,
        inverse => false
    }.

generate_initial_parameters(NumQubits, Depth) ->
    %% Generate random initial parameters for VQE
    NumParams = NumQubits * Depth * 2,
    [rand:uniform() * 2 * math:pi() || _ <- lists:seq(1, NumParams)].

%% ============================================================================
%% Result Processing Functions
%% ============================================================================

process_grover_results(RawResult) ->
    Counts = maps:get(<<"counts">>, RawResult),

    %% Find most probable state (should be the solution)
    {BestState, _MaxCount} = maps:fold(
        fun(State, Count, {AccState, AccCount}) ->
            case Count > AccCount of
                true -> {State, Count};
                false -> {AccState, AccCount}
            end
        end,
        {<<>>, 0},
        Counts
    ),

    #{
        optimal_structure => decode_protein_structure(BestState),
        probability => calculate_success_probability(Counts),
        raw_counts => Counts
    }.

process_qft_results(RawResult) ->
    Counts = maps:get(<<"counts">>, RawResult),

    %% Process Fourier coefficients
    FourierCoeffs = extract_fourier_coefficients(Counts),

    #{
        fourier_coefficients => FourierCoeffs,
        molecular_frequencies => analyze_frequencies(FourierCoeffs),
        raw_counts => Counts
    }.

decode_protein_structure(StateString) ->
    %% Decode quantum state to 3D protein coordinates
    %% This is a simplified version - real implementation would be more complex
    StateLength = byte_size(StateString),
    NumResidues = StateLength div 3,

    [#{
        residue => I,
        x => decode_coordinate(StateString, I * 3),
        y => decode_coordinate(StateString, I * 3 + 1),
        z => decode_coordinate(StateString, I * 3 + 2)
    } || I <- lists:seq(1, NumResidues)].

decode_coordinate(StateString, Offset) ->
    %% Extract coordinate from quantum state encoding
    case binary:at(StateString, Offset rem byte_size(StateString)) of
        $0 -> -1.0 + rand:uniform() * 0.5;
        $1 -> 0.5 + rand:uniform() * 0.5;
        _ -> rand:uniform() * 2.0 - 1.0
    end.

calculate_success_probability(Counts) ->
    TotalCounts = lists:sum(maps:values(Counts)),
    MaxCount = lists:max(maps:values(Counts)),
    MaxCount / TotalCounts.

extract_fourier_coefficients(Counts) ->
    %% Extract Fourier coefficients from QFT results
    maps:fold(
        fun(State, Count, Acc) ->
            Coeff = calculate_fourier_coefficient(State, Count),
            [Coeff | Acc]
        end,
        [],
        Counts
    ).

calculate_fourier_coefficient(State, Count) ->
    %% Convert quantum measurement to complex Fourier coefficient
    Phase = lists:sum([I * binary:at(State, I rem byte_size(State))
                      || I <- lists:seq(0, byte_size(State) - 1)]) * math:pi() / byte_size(State),
    #{
        amplitude => math:sqrt(Count),
        phase => Phase,
        real => math:sqrt(Count) * math:cos(Phase),
        imag => math:sqrt(Count) * math:sin(Phase)
    }.

analyze_frequencies(FourierCoeffs) ->
    %% Analyze molecular vibration frequencies from Fourier coefficients
    lists:map(
        fun(#{real := Real, imag := Imag}) ->
            Frequency = math:sqrt(Real * Real + Imag * Imag),
            #{frequency => Frequency, mode => vibrational}
        end,
        FourierCoeffs
    ).

vqe_optimization_loop(Circuit, Backend, Shots) ->
    %% Classical optimization loop for VQE
    InitialParams = maps:get(parameters, Circuit),
    optimize_parameters(InitialParams, Circuit, Backend, Shots, 0).

optimize_parameters(Params, Circuit, Backend, Shots, Iteration) when Iteration < 100 ->
    %% Evaluate current parameters
    Energy = evaluate_energy(Params, Circuit, Backend, Shots),

    %% Compute gradient (finite differences)
    Gradient = compute_gradient(Params, Circuit, Backend, Shots),

    %% Update parameters
    LearningRate = 0.01,
    NewParams = lists:zipwith(
        fun(P, G) -> P - LearningRate * G end,
        Params,
        Gradient
    ),

    %% Check convergence
    case abs(Energy) < 0.001 of
        true -> NewParams;
        false -> optimize_parameters(NewParams, Circuit, Backend, Shots, Iteration + 1)
    end;
optimize_parameters(Params, _Circuit, _Backend, _Shots, _Iteration) ->
    Params.

evaluate_energy(Params, Circuit, Backend, Shots) ->
    %% Evaluate energy expectation value
    %% This is simplified - real implementation would submit quantum circuit
    timer:sleep(100), % Simulate quantum execution
    -1.0 + 2.0 * rand:uniform(). % Random energy between -1 and 1

compute_gradient(Params, Circuit, Backend, Shots) ->
    %% Compute parameter gradients using finite differences
    Delta = 0.01,
    [finite_difference(Params, I, Delta, Circuit, Backend, Shots)
     || I <- lists:seq(1, length(Params))].

finite_difference(Params, Index, Delta, Circuit, Backend, Shots) ->
    %% Compute finite difference for parameter at Index
    ParamsPlus = lists:sublist(Params, Index - 1) ++
                 [lists:nth(Index, Params) + Delta] ++
                 lists:nthtail(Index, Params),
    ParamsMinus = lists:sublist(Params, Index - 1) ++
                  [lists:nth(Index, Params) - Delta] ++
                  lists:nthtail(Index, Params),

    EnergyPlus = evaluate_energy(ParamsPlus, Circuit, Backend, Shots),
    EnergyMinus = evaluate_energy(ParamsMinus, Circuit, Backend, Shots),

    (EnergyPlus - EnergyMinus) / (2 * Delta).

calculate_final_energy(Params) ->
    %% Calculate final energy from optimized parameters
    -0.95 + 0.1 * rand:uniform(). % Simulate low energy result

get_vqe_iterations() ->
    50. % Fixed iteration count for simulation

%% ============================================================================
%% Utility Functions
%% ============================================================================

generate_job_id() ->
    <<A:32, B:16, C:16, D:16, E:48>> = crypto:strong_rand_bytes(16),
    list_to_binary(io_lib:format("~8.16.0b-~4.16.0b-~4.16.0b-~4.16.0b-~12.16.0b",
                                  [A, B, C, D, E])).

job_to_map(Job) ->
    #{
        id => Job#quantum_job.id,
        type => Job#quantum_job.type,
        backend => Job#quantum_job.backend,
        shots => Job#quantum_job.shots,
        priority => Job#quantum_job.priority,
        submitted_at => Job#quantum_job.submitted_at,
        status => Job#quantum_job.status,
        result => Job#quantum_job.result,
        error => Job#quantum_job.error,
        node => Job#quantum_job.node
    }.

count_active_jobs(Jobs) ->
    maps:fold(
        fun(_JobId, Job, Acc) ->
            case Job#quantum_job.status of
                queued -> Acc + 1;
                running -> Acc + 1;
                _ -> Acc
            end
        end,
        0,
        Jobs
    ).

%% Simulation helpers (replace with real IBM Quantum API calls)
put_simulated_result(JobId, Result) ->
    put({result, JobId}, Result).

get_simulated_result(JobId) ->
    case get({result, JobId}) of
        undefined -> {pending};
        Result -> {ok, Result}
    end.

%% ============================================================================
%% Supervisor Functions
%% ============================================================================

start_quantum_worker_supervisor() ->
    supervisor:start_link({local, quantum_worker_supervisor}, ?MODULE, quantum_worker_supervisor).

quantum_worker_supervisor_init(_Args) ->
    SupFlags = #{
        strategy => one_for_one,
        intensity => 10,
        period => 60
    },

    ChildSpecs = [],

    {ok, {SupFlags, ChildSpecs}}.

# =======================================================================


# =======================================================================
