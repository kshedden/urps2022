using UnicodePlots, Distributions, GZip, CSV, DataFrames, Printf

# Generate uniformly distributed random points on the
# unit circle.
function circ_rand(ndyad, nday)
    u = 2*pi*rand(ndyad)
    x = kron(cos.(u), ones(nday))
    x += vcat([0.2*ar(nday, 0.95) for _ in 1:ndyad]...)
    y = kron(sin.(u), ones(nday))
    y += vcat([0.2*ar(nday, 0.95) for _ in 1:ndyad]...)
	return (x, y)
end

# Take the point x1+x2, y1+y2 and determine its angle
function circ_combine(x1, y1, x2, y2)
	x = x1 + x2
	y = y1 + y2
	return atan.(y, x)
end

# Generate an ar-1 process of length n with autocorrelation r.
function ar(n, r)
	e = zeros(n)
	e[1] = randn()
	f = sqrt(1 - r^2)
	for i in 2:n
		e[i] += r*e[i-1] + f*randn()
	end	
	return e
end

# Generate an independent ar-1 process for each dyad.
function genar(ndyad, nday, r)
	return vcat([ar(nday, r) for _ in 1:ndyad]...)
end

function gendat(offset, amplitude, level)
	@assert length(offset) == length(amplitude) == length(level)
	hr = zeros(length(offset), 1440)
	m = 1:1440
	for i in eachindex(offset)
		hr[i, :] = level[i] .+ amplitude[i]*sin.(2*pi*(m .+ offset[i])/1440)
		hr[i, :] .+= 10*ar(1440, 0.9)
	end
	return hr
end

# Number of minutes in a day
m = 1440

# Number of dyads
ndyad = 70

# Number of days per dyad
nday = 90


# HR level for patients and caregivers
ld = kron(randn(ndyad), ones(nday)) + genar(ndyad, nday, 0.8)
lp = kron(randn(ndyad), ones(nday)) + genar(ndyad, nday, 0.8)
lc = kron(randn(ndyad), ones(nday)) + genar(ndyad, nday, 0.8)
m, v = 80, 400
a, b = m^2 / v, v / m
lev_p = quantile(Gamma(a, b), cdf(Normal(0, 2), ld + lp))
lev_c = quantile(Gamma(a, b), cdf(Normal(0, 2), ld + lc))

# HR amplitude for patients and caregivers
ad = kron(randn(ndyad), ones(nday)) + genar(ndyad, nday, 0.8) 
ap = kron(randn(ndyad), ones(nday)) + genar(ndyad, nday, 0.8)
ac = kron(randn(ndyad), ones(nday)) + genar(ndyad, nday, 0.8)
m, v = 30, 15^2
a, b = m^2 / v, v / m
amp_p = quantile(Gamma(a, b), cdf(Normal(0, 2), ad + ap))
amp_c = quantile(Gamma(a, b), cdf(Normal(0, 2), ad + ac))

# HR offset for patients and caregivers
dx, dy = circ_rand(ndyad, nday)
x1, y1 = circ_rand(ndyad, nday)
x2, y2 = circ_rand(ndyad, nday)
off_p = 1440*circ_combine(dx, dy, x1, y1) / (2*pi)
off_c = 1440*circ_combine(dx, dy, x2, y2) / (2*pi)
 
hr_patient = gendat(off_p, amp_p, lev_p)
hr_caregiver = gendat(off_c, amp_c, lev_c)

function convert_df(x, ndyad, nday)

	x = DataFrame(x)
	c = [@sprintf("%d", m) for m in 1:1440]
	rename!(x, c)

    # Day indicator
    x[:, :day] = kron(ones(ndyad), 1:nday)

    # Person indicator
    x[:, :dyad] = kron(1:ndyad, ones(nday))

	v = ["dyad", "day"]
	push!(v, c...)
	x = x[:, v]

	x[!, :day] = Int.(x[:, :day])
	x[!, :dyad] = Int.(x[:, :dyad])

	return x
end

hr_patient = convert_df(hr_patient, ndyad, nday)
hr_caregiver = convert_df(hr_caregiver, ndyad, nday)

GZip.open("hr_patient.csv.gz", "w") do io
	CSV.write(io, hr_patient)
end

GZip.open("hr_caregiver.csv.gz", "w") do io
	CSV.write(io, hr_caregiver)
end
