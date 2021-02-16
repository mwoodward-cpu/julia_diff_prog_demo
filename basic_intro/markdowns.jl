using Weave

filename = normpath("weave.jmd")
weave(filename, out_path = :pwd)
