function save_election(election::Election, path::String)
	try
		jldsave(path; election)
	catch
		error("Could not save election to $(path)")
	end
end

function load_election(path::String)
	return load(path, "election")
end

function save_model(model::Abstract_model, path::String)
	try
		jldsave(path; model)
	catch
		error("Could not save model to $(path)")
	end
end

function load_model(path::String)
	return load(path, "model")
end

function save_config(config::Abstract_config, path::String)
	try
		jldsave(path; config)
	catch
		error("Could not save config to $(path)")
	end
end

function load_config(path::String)
	return load(path, "config")
end