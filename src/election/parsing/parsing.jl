function parse_data(data_path::String)
	ext = Symbol(lowercase(splitext(data_path)[2][2:end]))

	return parse_data(data_path, Val(ext))
end

parse_data(data_path::String, ext)::Election = throw(ArgumentError("Unsupported format of input data $ext. Supported: [toc, soi]"))