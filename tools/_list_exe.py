import sys
from PyInstaller.archive.readers import CArchiveReader

exe = sys.argv[1]
cr = CArchiveReader(exe)
names = []
try:
    for item in cr.contents():
        # item 可能是 (name, ...) 或 (name, (...))
        if isinstance(item, tuple):
            names.append(str(item[0]))
        else:
            names.append(str(item))
except Exception as e:
    print("contents() failed:", repr(e))
    try:
        toc = getattr(cr, "toc", None)
        if toc is not None:
            for k in toc:
                names.append(str(k))
    except Exception as e2:
        print("toc failed:", repr(e2))

keys = ("vcruntime", "msvcp", "concrt", "qt6", "qtcore", "qtwidgets",
        "api-ms", "python3", "python312", "python313", ".pyd")
print("=== matched binaries in exe ===")
for n in sorted(names):
    low = n.lower()
    if any(k in low for k in keys):
        print(n)
print("=== total entries:", len(names), "===")
