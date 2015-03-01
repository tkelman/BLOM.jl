functioncodes = Dict{Symbol, Float64}()
open(joinpath(dirname(@__FILE__), "BLOM_functioncodes.h")) do file
    for line in eachline(file)
        (func, val) = split(replace(line, "#define BLOM_FUNCTIONCODE_", ""))
        functioncodes[symbol(lowercase(func))] = float64(val)
    end
end
# min code value chosen large enough that (1+eps)^x overflows in Float64
minvaluemask = 0x4400_0000_0000_0000 # always flip these 2 bits on
minfunctioncode = reinterpret(Float64, minvaluemask) # 2.0^65
positivenonansmask = 0x7fef_ffff_ffff_ffff # always flip these 2 bits off
@assert functioncodes[:minval] == minfunctioncode

if WORD_SIZE == 64
    # only check against existing values on 64 bit,
    # since hashing returns UInt32 otherwise
    for f in symbolic_derivatives_1arg()
        newval = reinterpret(Float64, (hash(f[1]) | minvaluemask) &
            positivenonansmask)
        if !haskey(functioncodes, f[1])
            warn("new function $(f[1]) in Calculus package, tell " *
                "maintainer to run BLOM.regenerate_functioncodes()")
        elseif functioncodes[f[1]] != newval
            warn("function code for $(f[1]) did not match - saved value is ",
                functioncodes[f[1]], ", value from hash should be ", newval)
        end
    end
end

function regenerate_functioncodes()
    if WORD_SIZE != 64
        error("cannot regenerate BLOM_functioncodes.h on 32 bit platforms " *
            "due to different hashing results")
    end
    open(joinpath(dirname(@__FILE__), "BLOM_functioncodes.h"), "w") do file
        vals = Float64[]
        for f in symbolic_derivatives_1arg()
            newval = reinterpret(Float64, (hash(f[1]) | minvaluemask) &
                positivenonansmask)
            push!(vals, newval)
            fname = uppercase(string(f[1]))
            write(file, "#define BLOM_FUNCTIONCODE_$fname $newval\n")
            @assert (fname != "MINVAL") "encountered reserved name $(f[1])"
        end
        write(file, "#define BLOM_FUNCTIONCODE_MINVAL $minfunctioncode\n")
        if minimum(diff(sort!(vals))) == 0.0
            error("function names from Calculus package hashed too " *
                "close together")
        end
    end
end
