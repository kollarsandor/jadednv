# =======================================================================


defmodule ProteinAPI.MixProject do
  use Mix.Project

  def project do
    [
      app: :protein_api,
      version: "1.0.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases()
    ]
  end

  def application do
    [
      mod: {ProteinAPI.Application, []},
      extra_applications: [:logger, :runtime_tools, :crypto]
    ]
  end

  defp deps do
    [
      {:phoenix, "~> 1.7.0"},
      {:phoenix_live_view, "~> 0.20.0", optional: true},
      {:phoenix_pubsub, "~> 2.1"},
      {:plug_cowboy, "~> 2.6"},
      {:jason, "~> 1.4"},
      {:telemetry_metrics, "~> 0.6"},
      {:telemetry_poller, "~> 1.0"},
      {:cors_plug, "~> 3.0"}
    ]
  end

  defp aliases do
    [
      setup: ["deps.get"],
      "ecto.setup": [],
      "ecto.reset": [],
      test: ["test"],
      "assets.deploy": []
    ]
  end
end

# =======================================================================


# =======================================================================
