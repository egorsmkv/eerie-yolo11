import iree.compiler as _

lib_path = f'{_.__path__[0]}/_mlir_libs/'


with open('.cargo/config.toml', 'w') as f:
    f.write(f'''[build]
rustflags = ["-C", "link-arg=-Wl,-rpath,{lib_path}"]
rustdocflags = ["-C", "link-arg=-Wl,-rpath,{lib_path}"]

[env]
LIB_IREE_COMPILER = "{lib_path}"
''')
