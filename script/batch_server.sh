#!/usr/bin/env bash
    
usage() {
    echo "Usage: $(basename $0) [options]" >&2
    echo "Options:" >&2
    echo "  -f <config_file>    Configuration file path (default: config.yaml)" >&2
    echo "  -h, --help          Show this help message and exit" >&2
}

if [[ "$@" =~ "--help" ]] || [[ "$@" =~ "-h" ]]; then
    usage
    exit 0
fi

# move to project root directory
cd "$(dirname $0)/.." || { echo "Failed to change dicrectory to project root"; exit 1; }

if [ ! -d .venv ]; then
    echo "You need to create a virtual environment first."
    return 1
fi

source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
    
config_file="config.yaml"
while getopts "f:rh" opt; do
    case "$opt" in
        f)
            config_file=$OPTARG
            ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

python -O src/script/batch_server.py --config "$config_file"