import argparse
from app import create_app
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=os.environ.get("CONFIG_PATH", "configs/config.yaml"), help="Config path")
    args = parser.parse_args()

    app = create_app(args)
    app.run(debug=True)
