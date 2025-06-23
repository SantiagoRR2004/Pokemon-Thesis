# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_teams>"
  exit 1
fi

# Use the first argument as the number of iterations
count=$1

# Get the directory of the currently running script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for ((i=1; i<=count; i++))
do
  node "$SCRIPT_DIR/generate-team.js" > "$SCRIPT_DIR/randomTeam$i.txt"
done
