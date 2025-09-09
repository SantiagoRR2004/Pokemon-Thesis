import players
import inspect


# Get all classes defined in the players module
classes = inspect.getmembers(players, inspect.isclass)


print("=" * 50)

for name, obj in classes:
    if name != "AbstractAIPlayer":  # Skip the abstract base class
        print(f"Class {name} is valid with {obj.N_F_BATTLE} battle features.")

print("=" * 50)
