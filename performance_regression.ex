# =======================================================================


defmodule ProteinAPI.PerformanceRegression do
  @moduledoc """
  Performance regression CI with Benchee/pytest-benchmark/BenchmarkTools
  Automated performance testing and regression detection
  """

  use GenServer
  require Logger

  defstruct [
    :baseline_results,
    :current_results,
    :regression_threshold,
    :benchmark_history,
    :notification_handlers
  ]

  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def init(_) do
    state = %__MODULE__{
      baseline_results: load_baseline_results(),
      current_results: %{},
      regression_threshold: 0.1, # 10% regression threshold
      benchmark_history: :ets.new(:benchmark_history, [:ordered_set, :private]),
      notification_handlers: []
    }

    {:ok, state}
  end

  def run_benchmarks(suite_name \\ :all) do
    GenServer.call(__MODULE__, {:run_benchmarks, suite_name}, :timer.minutes(30))
  end

  def get_results(suite_name) do
    GenServer.call(__MODULE__, {:get_results, suite_name})
  end

  def set_baseline(suite_name) do
    GenServer.call(__MODULE__, {:set_baseline, suite_name})
  end

  def check_regressions do
    GenServer.call(__MODULE__, :check_regressions)
  end

  def handle_call({:run_benchmarks, suite_name}, _from, state) do
    results = case suite_name do
      :all -> run_all_benchmarks()
      :elixir -> run_elixir_benchmarks()
      :julia -> run_julia_benchmarks()
      :python -> run_python_benchmarks()
      specific -> run_specific_benchmark(specific)
    end

    timestamp = System.system_time(:second)
    :ets.insert(state.benchmark_history, {timestamp, suite_name, results})

    updated_state = %{state | current_results: Map.put(state.current_results, suite_name, results)}

    # Check for regressions
    regressions = detect_regressions(updated_state, suite_name)
    if not Enum.empty?(regressions) do
      notify_regressions(regressions, state.notification_handlers)
    end

    {:reply, {:ok, results}, updated_state}
  end

  def handle_call({:get_results, suite_name}, _from, state) do
    results = Map.get(state.current_results, suite_name, %{})
    {:reply, {:ok, results}, state}
  end

  def handle_call({:set_baseline, suite_name}, _from, state) do
    case Map.get(state.current_results, suite_name) do
      nil ->
        {:reply, {:error, :no_current_results}, state}
      results ->
        updated_baseline = Map.put(state.baseline_results, suite_name, results)
        save_baseline_results(updated_baseline)
        updated_state = %{state | baseline_results: updated_baseline}
        {:reply, :ok, updated_state}
    end
  end

  def handle_call(:check_regressions, _from, state) do
    all_regressions = Enum.flat_map(state.current_results, fn {suite_name, _results} ->
      detect_regressions(state, suite_name)
    end)

    {:reply, {:ok, all_regressions}, state}
  end

  defp run_all_benchmarks do
    %{
      elixir: run_elixir_benchmarks(),
      julia: run_julia_benchmarks(),
      python: run_python_benchmarks()
    }
  end

  defp run_elixir_benchmarks do
    Benchee.run(
      %{
        "protein_folding_full" => fn ->
          ProteinAPI.JuliaPort.fold_protein("MKFLVLLFNILCLFPVLAADNHGVGPQGASVILQTHDD", %{})
        end,
        "structure_validation" => fn ->
          structure = generate_test_structure()
          ProteinAPI.JuliaPort.validate_structure(structure)
        end,
        "confidence_computation" => fn ->
          structure = generate_test_structure()
          ProteinAPI.JuliaPort.compute_confidence(structure)
        end,
        "cache_read" => fn ->
          ProteinAPI.StructureCache.get("test_key")
        end,
        "cache_write" => fn ->
          ProteinAPI.StructureCache.put("test_key_#{:rand.uniform(1000)}", generate_test_structure())
        end,
        "job_submission" => fn ->
          job_data = %{
            type: "fold_protein",
            sequence: "MKFLVLLFNILCLFPVLAADNHGVGPQGAS",
            options: %{}
          }
          ProteinAPI.ComputationManager.submit_job(job_data)
        end,
        "http_request_processing" => fn ->
          conn = %Plug.Conn{
            method: "POST",
            request_path: "/api/fold",
            req_headers: [{"content-type", "application/json"}]
          }
          ProteinAPIWeb.ProteinController.fold_protein(conn, %{"sequence" => "MKFLVLLFNILCLFPVLAADNHGVG"})
        end
      },
      time: 10,
      memory_time: 2,
      reduction_time: 2,
      pre_check: true,
      formatters: [
        Benchee.Formatters.JSON
      ],
      save: [path: "benchmarks/elixir_results.json"]
    )
  end

  defp run_julia_benchmarks do
    # Run Julia BenchmarkTools benchmarks
    julia_script = """
    using BenchmarkTools
    using JSON3

    include("main.jl")

    # Warmup
    quantum_protein_folding_pipeline("MKFLVLLFNILCLFPVLAADNHGVGPQGAS")

    suite = BenchmarkGroup()

    # Core protein folding benchmarks
    suite["protein_folding"] = BenchmarkGroup()
    suite["protein_folding"]["short_sequence"] = @benchmarkable quantum_protein_folding_pipeline("MKFLVLLFNIL")
    suite["protein_folding"]["medium_sequence"] = @benchmarkable quantum_protein_folding_pipeline("MKFLVLLFNILCLFPVLAADNHGVGPQGAS")
    suite["protein_folding"]["long_sequence"] = @benchmarkable quantum_protein_folding_pipeline("MKFLVLLFNILCLFPVLAADNHGVGPQGASVILQTHDDGYMYPITMSISTDVSIPLASQKCYTGF")

    # AlphaFold3 model benchmarks
    suite["alphafold3"] = BenchmarkGroup()
    config = ModelConfig()
    model = Alphafold3(config)
    test_input = generate_test_input()

    suite["alphafold3"]["forward_pass"] = @benchmarkable $model($test_input)
    suite["alphafold3"]["attention"] = @benchmarkable $model.pairformer($test_input)
    suite["alphafold3"]["diffusion"] = @benchmarkable $model.diffusion($test_input)

    # Memory allocation benchmarks
    suite["memory"] = BenchmarkGroup()
    suite["memory"]["matrix_ops"] = @benchmarkable accelerated_distance_matrix(randn(Float32, 1000, 3))
    suite["memory"]["tensor_ops"] = @benchmarkable optimized_matmul(randn(Float32, 512, 512), randn(Float32, 512, 512))

    # Run benchmarks
    results = run(suite, verbose=true, seconds=10)

    # Convert to JSON-serializable format
    json_results = Dict()
    for (group_name, group) in results
        json_results[string(group_name)] = Dict()
        for (bench_name, bench_result) in group
            json_results[string(group_name)][string(bench_name)] = Dict(
                "time" => time(bench_result),
                "memory" => memory(bench_result),
                "allocs" => allocs(bench_result),
                "median_time" => median(bench_result.times),
                "mean_time" => mean(bench_result.times),
                "std_time" => std(bench_result.times)
            )
        end
    end

    # Save results
    open("benchmarks/julia_results.json", "w") do f
        JSON3.write(f, json_results)
    end

    println(JSON3.write(json_results))
    """

    {result, 0} = System.cmd("julia", ["-e", julia_script], stderr_to_stdout: true)

    case Jason.decode(result) do
      {:ok, parsed} -> parsed
      {:error, _} -> %{}
    end
  end

  defp run_python_benchmarks do
    # Run Python pytest-benchmark benchmarks
    python_script = """
import pytest
import json
import asyncio
import time
from quantum_integration import QuantumProteinFolder

class TestPerformance:
    @pytest.mark.benchmark(group="protein_folding")
    def test_fold_short_sequence(self, benchmark):
        folder = QuantumProteinFolder()
        sequence = "MKFLVLLFNIL"

        def fold():
            return asyncio.run(folder.fold_protein_async(sequence))

        result = benchmark(fold)
        assert result["status"] == "success"

    @pytest.mark.benchmark(group="protein_folding")
    def test_fold_medium_sequence(self, benchmark):
        folder = QuantumProteinFolder()
        sequence = "MKFLVLLFNILCLFPVLAADNHGVGPQGAS"

        def fold():
            return asyncio.run(folder.fold_protein_async(sequence))

        result = benchmark(fold)
        assert result["status"] == "success"

    @pytest.mark.benchmark(group="http_client")
    def test_http_request_performance(self, benchmark):
        import httpx

        def make_request():
            with httpx.Client() as client:
                response = client.post("http://localhost:4000/api/fold",
                                     json={"sequence": "MKFLVLLFNILCLFPVLAADNHGVG"})
                return response.status_code

        result = benchmark(make_request)
        assert result in [200, 202]

    @pytest.mark.benchmark(group="async_operations")
    def test_async_batch_processing(self, benchmark):
        folder = QuantumProteinFolder()
        sequences = ["MKFLVLLFNIL", "CLFPVLAADNH", "GVGPQGASVIL"] * 10

        async def batch_fold():
            tasks = [folder.fold_protein_async(seq) for seq in sequences]
            return await asyncio.gather(*tasks)

        def run_batch():
            return asyncio.run(batch_fold())

        results = benchmark(run_batch)
        assert len(results) == len(sequences)

if __name__ == "__main__":
    pytest.main([
        __file__,
        "--benchmark-json=benchmarks/python_results.json",
        "--benchmark-min-rounds=5",
        "--benchmark-warmup=on",
        "--benchmark-warmup-iterations=3"
    ])
"""

    File.write!("benchmark_runner.py", python_script)
    {result, exit_code} = System.cmd("python3", ["benchmark_runner.py"], stderr_to_stdout: true)

    if exit_code == 0 and File.exists?("benchmarks/python_results.json") do
      case File.read("benchmarks/python_results.json") do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, parsed} -> parsed
            {:error, _} -> %{}
          end
        {:error, _} -> %{}
      end
    else
      Logger.error("Python benchmarks failed: #{result}")
      %{}
    end
  end

  defp run_specific_benchmark(benchmark_name) do
    # Run a specific benchmark by name
    case benchmark_name do
      name when is_binary(name) ->
        # Custom benchmark execution logic
        %{}
      _ ->
        %{}
    end
  end

  defp detect_regressions(state, suite_name) do
    current = Map.get(state.current_results, suite_name, %{})
    baseline = Map.get(state.baseline_results, suite_name, %{})

    threshold = state.regression_threshold

    Enum.reduce(current, [], fn {benchmark_name, current_result}, acc ->
      case Map.get(baseline, benchmark_name) do
        nil -> acc # No baseline to compare against
        baseline_result ->
          regression = calculate_regression(baseline_result, current_result)
          if regression > threshold do
            [{suite_name, benchmark_name, regression, baseline_result, current_result} | acc]
          else
            acc
          end
      end
    end)
  end

  defp calculate_regression(baseline, current) do
    baseline_time = extract_benchmark_time(baseline)
    current_time = extract_benchmark_time(current)

    if baseline_time > 0 do
      (current_time - baseline_time) / baseline_time
    else
      0.0
    end
  end

  defp extract_benchmark_time(result) when is_map(result) do
    cond do
      Map.has_key?(result, "median_time") -> result["median_time"]
      Map.has_key?(result, "mean_time") -> result["mean_time"]
      Map.has_key?(result, "time") -> result["time"]
      Map.has_key?(result, :median) -> result.median
      Map.has_key?(result, :mean) -> result.mean
      true -> 0.0
    end
  end
  defp extract_benchmark_time(_), do: 0.0

  defp notify_regressions(regressions, handlers) do
    Enum.each(regressions, fn {suite, benchmark, regression, baseline, current} ->
      message = """
      Performance Regression Detected!

      Suite: #{suite}
      Benchmark: #{benchmark}
      Regression: #{Float.round(regression * 100, 2)}%
      Baseline: #{extract_benchmark_time(baseline)}
      Current: #{extract_benchmark_time(current)}
      """

      Logger.error(message)

      :telemetry.execute(
        [:protein_api, :performance, :regression_detected],
        %{regression_percentage: regression * 100},
        %{
          suite: suite,
          benchmark: benchmark,
          baseline_time: extract_benchmark_time(baseline),
          current_time: extract_benchmark_time(current)
        }
      )

      # Send notifications to configured handlers
      Enum.each(handlers, fn handler ->
        send(handler, {:regression_detected, suite, benchmark, regression})
      end)
    end)
  end

  defp generate_test_structure do
    %{
      "atoms" => [
        %{"x" => 1.0, "y" => 2.0, "z" => 3.0, "element" => "C"},
        %{"x" => 2.0, "y" => 3.0, "z" => 4.0, "element" => "N"},
        %{"x" => 3.0, "y" => 4.0, "z" => 5.0, "element" => "O"}
      ],
      "confidence" => [0.95, 0.92, 0.88]
    }
  end

  defp load_baseline_results do
    case File.read("benchmarks/baseline_results.json") do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, parsed} -> parsed
          {:error, _} -> %{}
        end
      {:error, _} -> %{}
    end
  end

  defp save_baseline_results(results) do
    File.mkdir_p!("benchmarks")
    content = Jason.encode!(results, pretty: true)
    File.write!("benchmarks/baseline_results.json", content)
  end
end

defmodule ProteinAPI.PerformanceRegression.CIRunner do
  @moduledoc """
  CI runner for automated performance regression testing
  """

  def run_ci_benchmarks do
    IO.puts("Starting CI performance benchmarks...")

    # Create benchmarks directory
    File.mkdir_p!("benchmarks")

    # Run all benchmark suites
    {:ok, results} = ProteinAPI.PerformanceRegression.run_benchmarks(:all)

    # Check for regressions
    {:ok, regressions} = ProteinAPI.PerformanceRegression.check_regressions()

    # Generate report
    report = generate_ci_report(results, regressions)
    File.write!("benchmarks/ci_report.json", Jason.encode!(report, pretty: true))

    # Print summary
    print_ci_summary(regressions)

    # Exit with error code if regressions found
    if not Enum.empty?(regressions) do
      System.halt(1)
    end
  end

  defp generate_ci_report(results, regressions) do
    %{
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      results: results,
      regressions: format_regressions(regressions),
      summary: %{
        total_benchmarks: count_benchmarks(results),
        regressions_found: length(regressions),
        status: if(Enum.empty?(regressions), do: "PASS", else: "FAIL")
      }
    }
  end

  defp format_regressions(regressions) do
    Enum.map(regressions, fn {suite, benchmark, regression, baseline, current} ->
      %{
        suite: suite,
        benchmark: benchmark,
        regression_percentage: Float.round(regression * 100, 2),
        baseline_time: ProteinAPI.PerformanceRegression.extract_benchmark_time(baseline),
        current_time: ProteinAPI.PerformanceRegression.extract_benchmark_time(current)
      }
    end)
  end

  defp count_benchmarks(results) do
    Enum.reduce(results, 0, fn {_suite, suite_results}, acc ->
      acc + map_size(suite_results)
    end)
  end

  defp print_ci_summary(regressions) do
    if Enum.empty?(regressions) do
      IO.puts("✅ All performance benchmarks passed!")
    else
      IO.puts("❌ Performance regressions detected:")
      Enum.each(regressions, fn {suite, benchmark, regression, _baseline, _current} ->
        IO.puts("  - #{suite}/#{benchmark}: #{Float.round(regression * 100, 2)}% slower")
      end)
    end
  end
end

# =======================================================================


# =======================================================================
