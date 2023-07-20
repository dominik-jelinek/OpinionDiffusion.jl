"""
	parse_data(data_path::String, ext::Val)::Election

Parses the data from the given path with the given extension.

# Arguments
- `data_path::String`: The path to the data to parse.

# Returns
- `election::Election`: The parsed election.
"""
function parse_data(data_path::String)
	ext = Symbol(lowercase(splitext(data_path)[2][2:end]))

	return parse_data(data_path, Val(ext))
end

parse_data(data_path::String, ext)::Election = throw(ArgumentError("Unsupported format of input data $ext. Supported: [toc, soi]"))
