function save_model(model::Abstract_model, path::String)
	try
		jldsave(path; model)
	catch
		mkpath(dirname(path))
		jldsave(path; model)
	end
end

function save_config(config::Abstract_config, path::String)
	try
		jldsave(path; config)
	catch
		mkpath(dirname(path))
		jldsave(path; config)
	end
end

function load_model(path::String)
	return load(path, "model")
end

function load_config(path::String)
	return load(path, "config")
end