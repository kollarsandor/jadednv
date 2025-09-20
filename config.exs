# =======================================================================


import Config

config :protein_api, ProteinAPIWeb.Endpoint,
  url: [host: "0.0.0.0", port: 4000],
  http: [ip: {0, 0, 0, 0}, port: 4000],
  render_errors: [
    formats: [json: ProteinAPIWeb.ErrorJSON]
  ],
  pubsub_server: ProteinAPI.PubSub,
  secret_key_base: "ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5ZXlKNGRpSTZJbFJzYlVsVWFWQmhOMmxyU0dNeE9UbGpPWEpJYzNWdFlYQjFNV0p3U0VFaWZRPT0="

config :protein_api,
  generators: [timestamp_type: :utc_datetime]

config :logger, level: :info

config :phoenix, :json_library, Jason

import_config "#{config_env()}.exs"

# =======================================================================


# =======================================================================
