import sys
import os

print("Python Executable:", sys.executable)
print("\nPython Version:", sys.version)
print("\nPython Path:")
for p in sys.path:
    print(f"  - {p}")

print("\nTrying to import lime...")
try:
    import lime
    print(f"SUCCESS! lime location: {lime.__file__}")
except ImportError as e:
    print(f"FAILED: {e}")
    
print("\nInstalled packages location:")
import site
print(site.getsitepackages())

