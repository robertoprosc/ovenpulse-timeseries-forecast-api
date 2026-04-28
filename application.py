import uvicorn
import argparse

from app import app

"""
    Application Entry Point: from here the service is started setting a specific port and host.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CLI parameters to run the bakery time-series anomaly and forecasting API'
    )

    parser.add_argument(
        '-port',
        dest='port',
        default=5003,
        type=int,
        action='store',
        help='API service port number')

    parser.add_argument(
        '-host',
        dest='host',
        default='127.0.0.1',
        type=str,
        action='store',
        help='API service host address')

    args = parser.parse_args()

    uvicorn.run('application:app', port=args.port, host=args.host)
