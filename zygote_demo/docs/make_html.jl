using Weave

filename = normpath("zygote_intro.jmd")
weave(filename, out_path = :pwd)
