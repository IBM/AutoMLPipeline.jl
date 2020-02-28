module ValDateFilters

using TSML.TSMLTypes
import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload
using TSML.Utils
using TSML.Imputers

using Dates
using DataFrames
using Statistics
using CSV
using CodecBzip2

using MLDataUtils: slidingwindow

export fit!,transform!

export Matrifier,Dateifier
export DateValizer,DateValgator,DateValNNer,DateValMultiNNer
export CSVDateValReader, CSVDateValWriter, DateValLinearImputer
export BzCSVDateValReader

const gAggDict = Dict(
    :median => Statistics.median,
    :mean =>   Statistics.mean,
    :maximum => Statistics.maximum,
    :minimum => Statistics.minimum,
    :sum => sum
)

"""
    Matrifier(Dict(
       Dict(
        :ahead => 1,
        :size => 7,
        :stride => 1,
      )
    )

Converts a 1-D timeseries into sliding window matrix for ML training:
- `:ahead` => steps ahead to predict
- `:size` => size of sliding window
- `:stride` => amount of overlap in sliding window

Example:

    mtr = Matrifier(Dict(:ahead=>24,:size=>24,:stride=>5))
    lower = DateTime(2017,1,1)
    upper = DateTime(2017,1,5)
    dat=lower:Dates.Hour(1):upper |> collect
    vals = 1:length(dat)
    x = DataFrame(Date=dat,Value=vals)
    fit!(mtr,x)
    res = transform!(mtr,x)


Implements: `fit!`, `transform`
"""
mutable struct Matrifier <: Transformer
  model
  args

  function Matrifier(args=Dict())
    default_args = Dict{Symbol,Any}(
        :ahead => 1,
        :size => 7,
        :stride => 1,
    )
    new(nothing,mergedict(default_args,args))
  end
end


"""
    fit!(mtr::Matrifier,xx::T,y::Vector=Vector()) where {T<:Union{Matrix,Vector,DataFrame}}

Checks and validate inputs are in correct structure
"""
function fit!(mtr::Matrifier,xx::DataFrame,y::Vector=[]) 
  x = deepcopy(xx.Value) |> collect
  x isa Vector || error("data should be a vector")
  mtr.model = mtr.args
end

"""
    transform!(mtr::Matrifier,xx::T) where {T<:Union{Matrix,Vector,DataFrame}}

Applies the parameters of sliding windows to create the corresponding matrix
"""
function transform!(mtr::Matrifier,xx::DataFrame)
  x = deepcopy(xx.Value) |> collect
  x isa Vector || error("data should be a vector")
  mtype = eltype(x)
  res=toMatrix(mtr,x)
  resarray=convert(Array{mtype},res) |> DataFrame
  rename!(resarray,names(resarray)[end] => :output)
end

function toMatrix(mtr::Transformer, x::Vector)
  stride=mtr.args[:stride];sz=mtr.args[:size];ahead=mtr.args[:ahead]
  @assert stride>0 && sz>0 && ahead > 0
  xlength = length(x)
  xlength > sz || error("data too short for the given size of sliding window")
  ndx=collect(xlength:-1:1)
  mtuples = slidingwindow(i->(i-ahead),ndx,sz,stride)
  height=size(mtuples)[1]
  mmatrix = Array{Union{DateTime,<:Real},2}(zeros(height,sz+1))
  ctr=1
  gap = xlength - mtuples[1][2][1]
  for (s,k) in mtuples
    v = [reverse(s);k] .+ gap
    mmatrix[ctr,:].=x[v]
    ctr+=1
  end
  mmatrix
end

### ====


"""
    Dateifier(args=Dict())
       Dict(
        :ahead => 1,
        :size => 7,
        :stride => 1
       )
    )

Converts a 1-D date series into sliding window matrix for ML training

Example: 

    dtr = Dateifier(Dict())
    lower = DateTime(2017,1,1)
    upper = DateTime(2018,1,31)
    dat=lower:Dates.Day(1):upper |> collect
    vals = rand(length(dat))
    x=DataFrame(Date=dat,Value=vals)
    fit!(dtr,x)
    res = transform!(dtr,x)

Implements: `'fit!`, `transform!`
"""
mutable struct Dateifier <: Transformer
  model
  args

  function Dateifier(args=Dict())
    default_args = Dict{Symbol,Any}(
        :ahead => 1,
        :size => 7,
        :stride => 1
    )
    new(nothing,mergedict(default_args,args))
  end
end

"""
    fit!(dtr::Dateifier,xx::T,y::Vector=[]) where {T<:Union{Matrix,Vector,DataFrame}}

Computes range of dates to be used during transform.
"""
function fit!(dtr::Dateifier,xx::DataFrame,y::Vector=[])
  x = deepcopy(xx.Date)
  (eltype(x) <: DateTime || eltype(x) <: Date) || error("array element types are not dates")
  dtr.args[:lower] = minimum(x)
  dtr.args[:upper] = maximum(x)
  dtr.model = dtr.args
end

"""
    transform!(dtr::Dateifier,xx::T) where {T<:Union{Matrix,Vector,DataFrame}}

Transforms to day of the month, day of the week, etc
"""
function transform!(dtr::Dateifier,xx::DataFrame)
  x = deepcopy(xx.Date)
  x isa Vector || error("data should be a vector")
  @assert eltype(x) <: DateTime || eltype(x) <: Date
  res=toMatrix(dtr,x)
  endpoints = convert(Array{DateTime},res)[:,end-1]
  dt = DataFrame()
  dt.year=Dates.year.(endpoints)
  dt.month=Dates.month.(endpoints)
  dt.day=Dates.day.(endpoints)
  dt.hour=Dates.hour.(endpoints)
  dt.week=Dates.week.(endpoints)
  dt.dow=Dates.dayofweek.(endpoints)
  dt.doq=Dates.dayofquarter.(endpoints)
  dt.qoy=Dates.quarterofyear.(endpoints)
  dtr.args[:header] = names(dt)
  return dt
end


"""
    DateValgator(args=Dict())
       Dict(
        :dateinterval => Dates.Hour(1),
        :aggregator => :median
      )
    )

Aggregates values based on date period specified.

Example:

    # generate random values with missing data
    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
    gval = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gmissing = 50000
    gndxmissing = Random.shuffle(1:length(gdate))[1:gmissing]
    X = DataFrame(Date=gdate,Value=gval)
    X.Value[gndxmissing] .= missing

    dtvlmean = DateValgator(Dict(
          :dateinterval=>Dates.Hour(1),
          :aggregator => :mean))
    fit!(dtvlmean,X)
    res = transform!(dtvlmean,X)

Implements: `fit!`, `transform!`
"""
mutable struct DateValgator <: Transformer
  model
  args
  function DateValgator(args=Dict())
    default_args = Dict{Symbol,Any}(
        :dateinterval => Dates.Hour(1),
        :aggregator => :median
    )
    new(nothing,mergedict(default_args,args))
  end
end

function validdateval!(x::DataFrame)
  size(x)[2] == 2 || error("Date Val timeseries need two columns")
  (eltype(x[:,1]) <: DateTime || eltype(x[:,1]) <: Date) || error("array element types are not dates")
  eltype(x[:,2]) <: Union{Missing,Real} || error("array element types are not values")
  cnames = names(x)
  rename!(x,Dict(cnames[1]=>:Date,cnames[2]=>:Value))
end

"""
    fit!(dvmr::DateValgator,xx::T,y::Vector=[]) where {T<:Union{Matrix,DataFrame}}

Checks and validates arguments.
"""
function fit!(dvmr::DateValgator,xx::DataFrame,y::Vector=[])
  x = deepcopy(xx)
  validdateval!(x)
  aggr = dvmr.args[:aggregator] 
  aggr in keys(gAggDict) || error("aggregator function passed is not recognized: ",aggr)
  dvmr.model=dvmr.args
end

"""
    transform!(dvmr::DateValgator,xx::T) where {T<:DataFrame}

Aggregates values grouped by date-time period using aggregate 
function such as mean, median, maximum, minimum. Default is mean.
"""
function transform!(dvmr::DateValgator,xx::DataFrame)
  x = deepcopy(xx)
  validdateval!(x)
  # make sure aggregator function exists
  aggr = dvmr.args[:aggregator] 
  aggr in keys(gAggDict) || error("aggregator function passed is not recognized: ",aggr)
  # get the Statistics function
  aggfn = gAggDict[aggr]
  # pass the aggregator function to the generic aggregator function
  fn = aggregatorclskipmissing(aggfn)
  grpby = typeof(dvmr.args[:dateinterval])
  sym = Symbol(grpby)
  x[!,sym] = round.(x.Date,grpby)
  aggr=by(x,sym,MeanValue = :Value=>fn)
  rename!(aggr,Dict(names(aggr)[1]=>:Date,names(aggr)[2]=>:Value))
  lower = round(minimum(x.Date),grpby)
  upper = round(maximum(x.Date),grpby)
  #create list of complete dates and join with aggregated data
  cdate = DataFrame(Date = collect(lower:dvmr.args[:dateinterval]:upper))
  joined = join(cdate,aggr,on=:Date,kind=:left)
  joined
end

"""
    DateValizer(
       Dict(
        :medians => DataFrame(),
        :dateinterval => Dates.Hour(1)
      )
    )

Normalizes and cleans time series by replacing `missings` with global medians 
computed based on time period groupings.

Example:

    # generate random values with missing data
    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
    gval = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gmissing = 50000
    gndxmissing = Random.shuffle(1:length(gdate))[1:gmissing]
    X = DataFrame(Date=gdate,Value=gval)
    X.Value[gndxmissing] .= missing

    dvzr = DateValizer(Dict(:dateinterval=>Dates.Hour(1)))
    fit!(dvzr,X)
    transform!(dvzr,X)


Implements: `fit!`, `transform!`
"""
mutable struct DateValizer <: Transformer
  model
  args

  function DateValizer(args=Dict())
    default_args = Dict{Symbol,Any}(
        :medians => DataFrame(),
        :dateinterval => Dates.Hour(1)
    )
    new(nothing,mergedict(default_args,args))
  end
end

function getMedian(t::Type{T},xx::DataFrame) where {T<:Union{TimePeriod,DatePeriod}}
  x = deepcopy(xx)
  sgp = Symbol(t)
  fn = Dict(Dates.Second=>Dates.second,
            Dates.Minute=>Dates.minute,
            Dates.Hour=>Dates.hour,
            Dates.Day=>Dates.day,
            Dates.Month=>Dates.month)
  try
    x[!,sgp]=fn[t].(x.Date)
  catch
    error("unknown dateinterval")
  end
  gpmeans = by(x,sgp,Value = :Value => skipmedian)
  gpmeans
end

function fullaggregate!(dvzr::DateValizer,xx::DataFrame)
  x = deepcopy(xx)
  grpby = typeof(dvzr.args[:dateinterval])
  sym = Symbol(grpby)
  x[!,sym] = round.(x.Date,grpby)
  aggr = by(x,sym,MeanValue = :Value=>skipmedian)
  rename!(aggr,Dict(names(aggr)[1]=>:Date,names(aggr)[2]=>:Value))
  lower = minimum(x.Date)
  upper = maximum(x.Date)
  #create list of complete dates and join with aggregated data
  cdate = DataFrame(Date = collect(lower:dvzr.args[:dateinterval]:upper))
  joined = join(cdate,aggr,on=:Date,kind=:left)
  joined
end

"""
    fit!(dvzr::DateValizer,xx::T,y::Vector=[]) where {T<:DataFrame}

Validates input and computes global medians grouped by time period.
"""
function fit!(dvzr::DateValizer,xx::DataFrame,y::Vector=[]) 
  x = deepcopy(xx)
  validdateval!(x)
  # get complete dates and aggregate
  joined = fullaggregate!(dvzr,x)
  grpby = typeof(dvzr.args[:dateinterval])
  sym = Symbol(grpby)
  medians = getMedian(grpby,joined)
  dvzr.args[:medians] = medians
  dvzr.model=dvzr.args
end

"""
    transform!(dvzr::DateValizer,xx::T) where {T<:DataFrame}

Replaces `missing` with the corresponding global medians with respect to time period.
"""
function transform!(dvzr::DateValizer,xx::DataFrame) 
  x = deepcopy(xx)
  validdateval!(x)
  # get complete dates, aggregate, and get medians
  joined = fullaggregate!(dvzr,x)
  # copy medians
  medians = dvzr.args[:medians]
  grpby = typeof(dvzr.args[:dateinterval])
  sym = Symbol(grpby)
  fn = Dict(Dates.Hour=>Dates.hour,
            Dates.Minute=>Dates.minute,
            Dates.Second=>Dates.second,
            Dates.Day => Dates.day,
            Dates.Month=>Dates.month)
  try
    joined[!,sym]=fn[grpby].(joined.Date)
  catch
    error("unknown dateinterval")
  end
  # find indices of missing
  missingndx = findall(ismissing.(joined.Value))
  jmndx=joined[missingndx,sym] .+ 1 # get time period index of missing, convert 0 index time to 1 index
  missingvals::SubArray = @view joined[missingndx,:Value]
  missingvals .= medians[jmndx,:Value] # replace missing with median value
  sum(ismissing.(joined.Value)) == 0 || error("Aggregation by time period failed to replace missings")
  joined[:,[:Date,:Value]]
end

"""
    DateValNNer(
       Dict(
          :missdirection => :symmetric, #:reverse, # or :forward or :symmetric
          :dateinterval => Dates.Hour(1),
          :nnsize => 1,
          :strict => true,
          :aggregator => :median
      )
    )
 

Fills `missings` with their nearest-neighbors.
- `:missdirection` => direction to fill missing data (:symmetric, :reverse, :forward) 
- `:dateinterval` => time period to use for grouping,
- `:nnsize` => neighborhood size,
- `:strict` => boolean value to indicate whether to be strict about replacement or not,
- `:aggregator => function to aggregate based on date interval

Example:

    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
    gval = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gmissing = 50000
    gndxmissing = Random.shuffle(1:length(gdate))[1:gmissing]
    X = DataFrame(Date=gdate,Value=gval)
    X.Value[gndxmissing] .= missing

    dnnr = DateValNNer(Dict(
          :dateinterval=>Dates.Hour(1),
          :nnsize=>10,
          :missdirection => :symmetric,
          :strict=>true,
          :aggregator => :mean))
    fit!(dnnr,X)
    transform!(dnnr,X)

 
Implements: `fit!`, transform!`
"""
mutable struct DateValNNer <: Transformer
  model
  args

  function DateValNNer(args=Dict())
    default_args = Dict{Symbol,Any}(
        :missdirection => :symmetric, #:reverse, # or :forward or :symmetric
        :dateinterval => Dates.Hour(1),
        :nnsize => 1,
        :strict => true,
        :aggregator => :median
    )
    new(nothing,mergedict(default_args,args))
  end
end

"""
    fit!(dnnr::DateValNNer,xx::T,y::Vector=[]) where {T<:DataFrame}

Validates and checks arguments for errors.
"""
function fit!(dnnr::DateValNNer,xx::DataFrame,y::Vector=[])
  x = deepcopy(xx)
  validdateval!(x)
  aggr = dnnr.args[:aggregator]
  aggr in keys(gAggDict) || error("aggregator function passed is not recognized: ",aggr)
  dnnr.model=dnnr.args
end

"""
    transform!(dnnr::DateValNNer,xx::T) where {T<:DataFrame}

Replaces `missings` by nearest neighbor looping over the dataset until all missing values are gone.
"""
function transform!(dnnr::DateValNNer,xx::DataFrame)
  x = deepcopy(xx)
  validdateval!(x)
  # make sure aggregator function exists
  aggr = dnnr.args[:aggregator]
  aggr in keys(gAggDict) || error("aggregator function pass is not recognized: ",aggr)
  # get the Statistics function
  aggfn = gAggDict[aggr]
  # pass the aggregator function to the generic aggregator function
  fn = aggregatorclskipmissing(aggfn)
  grpby = typeof(dnnr.args[:dateinterval])
  sym = Symbol(grpby)
  # aggregate by time period
  x[!,sym] = round.(x.Date,grpby)
  aggr = by(x,sym,MeanValue = :Value=>fn)
  rename!(aggr,Dict(names(aggr)[1]=>:Date,names(aggr)[2]=>:Value))
  lower = round(minimum(x.Date),grpby)
  upper = round(maximum(x.Date),grpby)
  #create list of complete dates and join with aggregated data
  cdate = DataFrame(Date = collect(lower:dnnr.args[:dateinterval]:upper))
  joined = join(cdate,aggr,on=:Date,kind=:left)
  missingcount = sum(ismissing.(joined.Value))
  dnnr.args[:missingcount] = missingcount
  res = transform_worker!(dnnr,joined)
  count=1
  if dnnr.args[:missdirection] == :symmetric
    while sum(ismissing.(res.Value)) > 0
      res = transform_worker!(dnnr,res)
      count += 1
    end
  end
  dnnr.args[:loopcount] = count
  res
end

function transform_worker!(dnnr::DateValNNer,joinc::DataFrame)
  joined = deepcopy(joinc)
  maxrow = size(joined)[1]

  # to fill-in with nearest neighbors
  nnsize::Int64 = dnnr.args[:nnsize]
  themissing = findall(ismissing.(joined.Value))
  # ==== symmetric nearest neighbor
  missingndx = DataFrame()
  if dnnr.args[:missdirection] == :symmetric
    missed = themissing |> reverse
    missingndx.Missed = missed
    # get lower:upper range
    missingndx.neighbors = map(missingndx.Missed) do m
      lower = (m-nnsize >= 1) ? (m-nnsize) : 1
      upper = (m+nnsize <= maxrow) ? m+nnsize : maxrow
      lower:upper
    end
  else
    # ===== reverse and forward
    missed = (dnnr.args[:missdirection] == :reverse) ? (themissing |> reverse) : themissing
    missingndx.Missed = missed
    # dealing with boundary exceptions, default to range until the maxrow
    missingndx.neighbors = (m->((m+1>=maxrow) || (m+nnsize>=maxrow)) ? (m+1:maxrow) : (m+1:m+nnsize)).(missingndx.Missed) # NN ranges
  end
  #joined[missingndx[:Missed],:Value] = (r -> skipmedian(joined[r,:Value])).(missingndx[:neighbors]) # iterate to each range
  missingvals::SubArray = @view joined[missingndx.Missed,:Value] # get view of only missings
  missingvals .=  (r -> skipmedian(joined[r,:Value])).(missingndx.neighbors) # replace with nn medians
  dnnr.args[:strict] && (sum(ismissing.(joined.Value)) == 0 || error("Nearest Neigbour algo failed to replace missings"))
  joined
end

"""
    CSVDateValReader(
       Dict(
          :filename => "",
          :dateformat => ""
       )
    )

Reads csv file and parse date using the given format.
- `:filename` => complete path including filename of csv file
- `:dateformat` => date format to parse

Example:

    inputfile =joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
    csvreader = CSVDateValReader(Dict(:filename=>inputfile,:dateformat=>"d/m/y H:M"))
    fit!(csvreader)
    df = transform!(csvreader)

    # using pipeline workflow
    filter1 = DateValgator()
    filter2 = DateValNNer(Dict(:nnsize=>1))
    mypipeline = Pipeline(Dict(
          :transformers => [csvreader,filter1,filter2]
      )
    )
    fit!(mypipeline)
    res=transform!(mypipeline)


Implements: `fit!`, `transform!`
"""
mutable struct CSVDateValReader <: Transformer
    model
    args
    function CSVDateValReader(args=Dict())
        default_args = Dict(
            :filename => "",
            :dateformat => ""
        )
        new(nothing,mergedict(default_args,args))
    end
end

"""
    fit!(csvrdr::CSVDateValReader,x::T=[],y::Vector=[]) where {T<:Union{DataFrame,Vector,Matrix}}

Makes sure filename and dateformat are not empty strings.
"""
function fit!(csvrdr::CSVDateValReader,x::DataFrame=DataFrame(),y::Vector=[])
    fname = csvrdr.args[:filename]
    fmt = csvrdr.args[:dateformat]
    (fname != "" && fmt != "") || error("missing filename or date format")
    csvrdr.model = csvrdr.args
end

"""
    transform!(csvrdr::CSVDateValReader,x::T=[]) where {T<:Union{DataFrame,Vector,Matrix}}

Uses CSV package to read the csv file and converts it to dataframe.
"""
function transform!(csvrdr::CSVDateValReader,x::DataFrame=DataFrame())
    fname = csvrdr.args[:filename]
    fmt = csvrdr.args[:dateformat]
    df = CSV.read(fname) |> DataFrame
    ncol(df) == 2 || error("dataframe should have only two columns: Date,Value")
    rename!(df,names(df)[1]=>:Date,names(df)[2]=>:Value)
    if !(eltype(df.Date) <: DateTime )
      df.Date = DateTime.(df.Date,fmt)
    end
    df
end

"""
    CSVDateValWriter(
       Dict(
          :filename => "",
          :dateformat => ""
       )
    )

Writes the time series dataframe into a file with the given date format.

Example:

    inputfile =joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
    outputfile = joinpath("/tmp/test.csv")
    csvreader = CSVDateValReader(Dict(:filename=>inputfile,:dateformat=>"d/m/y H:M"))
    csvwtr = CSVDateValWriter(Dict(:filename=>outputfile,:dateformat=>"d/m/y H:M"))
    filter1 = DateValgator()
    filter2 = DateValNNer(Dict(:nnsize=>1))
    mypipeline = Pipeline(Dict(
          :transformers => [csvreader,filter1,filter2,csvwtr]
      )
    )
    fit!(mypipeline)
    res=transform!(mypipeline)

    # read back what was written to validate
    csvreader = CSVDateValReader(Dict(:filename=>outputfile,:dateformat=>"y-m-d HH:MM:SS"))
    fit!(csvreader)
    transform!(csvreader)

Implements: `fit!`, `transform!`
"""
mutable struct CSVDateValWriter <: Transformer
    model
    args
    function CSVDateValWriter(args=Dict())
        default_args = Dict(
            :filename => "",
            :dateformat => ""
        )
        new(nothing,mergedict(default_args,args))
    end
end

"""
    fit!(csvwtr::CSVDateValWriter,x::T=[],y::Vector=[]) where {T<:Union{DataFrame,Vector,Matrix}}

Makes sure filename and dateformat are not empty strings.
"""
function fit!(csvwtr::CSVDateValWriter,x::DataFrame=DataFrame(),y::Vector=[])
    fname = csvwtr.args[:filename]
    fmt = csvwtr.args[:dateformat]
    fname != ""  || error("missing filename")
    csvwtr.model = csvwtr.args
end

"""
    transform!(csvwtr::CSVDateValWriter,x::T) where {T<:Union{DataFrame,Vector,Matrix}}

Uses CSV package to write the dataframe into a csv file.
"""
function transform!(csvwtr::CSVDateValWriter,x::DataFrame)
    fname = csvwtr.args[:filename]
    fmt = csvwtr.args[:dateformat]
    df = deepcopy(x) |> DataFrame
    if ncol(df) == 2 
      rename!(df,names(df)[1]=>:Date,names(df)[2]=>:Value)
      eltype(df.Date) <: DateTime || error("Date format error")
    end
    df |> CSV.write(fname)
    return df
end

"""
    BzCSVDateValReader(
       Dict(
          :filename => "",
          :dateformat => ""
       )
    )

Reads Bzipped csv file and parse date using the given format.
- `:filename` => complete path including filename of csv file
- `:dateformat` => date format to parse

Example:

    inputfile =joinpath(dirname(pathof(TSML)),"../data/testdata.csv.bz2")
    csvreader = BzCSVDateValReader(Dict(:filename=>inputfile,:dateformat=>"d/m/y H:M"))
    filter1 = DateValgator()
    filter2 = DateValNNer(Dict(:nnsize=>1))
    mypipeline = Pipeline(Dict(
          :transformers => [csvreader,filter1,filter2]
      )
    )
    fit!(mypipeline)
    res=transform!(mypipeline)

Implements: `fit!`, `transform!`
"""
mutable struct BzCSVDateValReader <: Transformer
    model
    args
    function BzCSVDateValReader(args=Dict())
        default_args = Dict(
            :filename => "",
            :dateformat => ""
        )
        new(nothing,mergedict(default_args,args))
    end
end

"""
    fit!(bzcsvrdr::BzCSVDateValReader,x::T=[],y::Vector=[]) where {T<:Union{DataFrame,Vector,Matrix}}

Makes sure filename and dateformat are not empty strings.
"""
function fit!(bzcsvrdr::BzCSVDateValReader,x::DataFrame=DataFrame(),y::Vector=[])
    fname = bzcsvrdr.args[:filename]
    fmt = bzcsvrdr.args[:dateformat]
    (fname != "" && fmt != "") || error("missing filename or date format")
    bzcsvrdr.model = bzcsvrdr.args
end

"""
    transform!(bzcsvrdr::BzCSVDateValReader,x::T=[]) where {T<:Union{DataFrame,Vector,Matrix}}

Uses CodecBzip2 package to read the csv file and converts it to dataframe.
"""
function transform!(bzcsvrdr::BzCSVDateValReader,x::DataFrame=DataFrame())
    fname = bzcsvrdr.args[:filename]
    fmt = bzcsvrdr.args[:dateformat]
    stream = Bzip2DecompressorStream(open(fname))
    df = CSV.read(stream) |> DataFrame
    ncol(df) == 2 || error("dataframe should have only two columns: Date,Value")
    rename!(df,names(df)[1]=>:Date,names(df)[2]=>:Value)
    df.Date = DateTime.(df.Date,fmt)
    df
end


"""
    DateValMultiNNer(
       Dict(
          :type => :knn # :linear
          :missdirection => :symmetric, #:reverse, # or :forward or :symmetric
          :dateinterval => Dates.Hour(1),
          :nnsize => 1,
          :strict => true,
          :aggregator => :median
      )
    )
 

Fills `missings` with their nearest-neighbors. It assumes that first column is a Date class
and the other columns are Union{Missings,Real}. It uses DateValNNer and DateValizer+Impute to
process each numeric column concatendate with the Date column.
- `:type` => type of imputation which can be a linear interpolation or nearest neighbor
- `:missdirection` => direction to fill missing data (:symmetric, :reverse, :forward) 
- `:dateinterval` => time period to use for grouping,
- `:nnsize` => neighborhood size,
- `:strict` => boolean value to indicate whether to be strict about replacement or not,
- `:aggregator => function to aggregate based on date interval

Example:

    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
    gval1 = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gval2 = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gval3 = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gmissing = 50000
    gndxmissing1 = Random.shuffle(1:length(gdate))[1:gmissing]
    gndxmissing2 = Random.shuffle(1:length(gdate))[1:gmissing]
    gndxmissing3 = Random.shuffle(1:length(gdate))[1:gmissing]
    X = DataFrame(Date=gdate,Temperature=gval1,Humidity=gval2,Ozone=gval3)
    X.Temperature[gndxmissing1] .= missing
    X.Humidity[gndxmissing2] .= missing
    X.Ozone[gndxmissing3] .= missing

    dnnr = DateValMultiNNer(Dict(
          :type=>:linear,
          :dateinterval=>Dates.Hour(1),
          :nnsize=>10,
          :missdirection => :symmetric,
          :strict=>true,
          :aggregator => :mean))
    fit!(dnnr,X)
    transform!(dnnr,X)

 
Implements: `fit!`, transform!`
"""
mutable struct DateValMultiNNer <: Transformer
  model
  args

  function DateValMultiNNer(args=Dict())
    default_args = Dict{Symbol,Any}(
        :type => :knn,
        :missdirection => :symmetric, #:reverse, # or :forward or :symmetric
        :dateinterval => Dates.Hour(1),
        :nnsize => 1,
        :strict => true,
        :aggregator => :median
    )
    new(nothing,mergedict(default_args,args))
  end
end

function multivalidateval(x::DataFrame)
  #size(x)[2] > 2 || error("Multi Date Val timeseries need more than two columns")
  (eltype(x[:,1]) <: DateTime || eltype(x[:,1]) <: Date) || error("array element types are not dates")
  sum(broadcast(y->eltype(y)<:Union{Missing,Real},eachcol(x[:,2:end]))) == ncol(x)-1 || error("columns 2:end should be real numbers")
end

"""
    fit!(dnnr::DateValMultiNNer,xx::T,y::Vector=[]) where {T<:DataFrame}

Validates and checks arguments for errors.
"""
function fit!(dnnr::DateValMultiNNer,xx::DataFrame,y::Vector=[])
  x = deepcopy(xx)
  # validate it's multi-dimensional and first column is date
  multivalidateval(x)
  cnames = names(x)
  rename!(x,Dict(cnames[1]=>:Date))
  aggr = dnnr.args[:aggregator]
  aggr in keys(gAggDict) || error("aggregator function passed is not recognized: ",aggr)
  dnnr.model=dnnr.args
end

"""
    transform!(dnnr::DateValMultiNNer,xx::T) where {T<:DataFrame}

Replaces `missings` by nearest neighbor or linear interpolation by looping over the dataset 
for each column until all missing values are gone.
"""
function transform!(dnnr::DateValMultiNNer,xx::DataFrame)
  x = deepcopy(xx)
  # make sure data is valid
  multivalidateval(x)
  # loop each column and call ValdateNNer to impute
  df = DataFrame()
  if dnnr.args[:type] == :knn
    df = knnimpute(dnnr,x)
  elseif dnnr.args[:type] == :linear
    df = linearimpute(dnnr,x)
  end
  return df
end

function knnimpute(dnnr::DateValMultiNNer,x::DataFrame)
  valnner = DateValNNer(dnnr.args)
  df = DataFrame(Date=x.Date) 
  cnames = names(x)
  for y in eachcol(x[:,2:end])
    input = DataFrame(Date=x.Date,Value=y)
    fit!(valnner,input)
    res=transform!(valnner,input)
    df = join(df,res,on=:Date,makeunique=true)
  end
  rename!(df,cnames)
  return df
end

function linearimpute(dnnr::DateValMultiNNer,x::DataFrame)
  valgator = DateValgator(dnnr.args)
  linearputer = DateValLinearImputer(dnnr.args)
  df = DataFrame(Date=x.Date) 
  cnames = names(x)
  for y in eachcol(x[:,2:end])
    input = DataFrame(Date=x.Date,Value=y)
    fit!(valgator,input)
    agg=transform!(valgator,input)
    fit!(linearputer,agg)
    res=transform!(linearputer,agg)
    df = join(df,res,on=:Date,makeunique=true)
  end
  rename!(df,cnames)
  return df
end


"""
    DateValLinearImputer(
       Dict(
          :dateinterval => Dates.Hour(1),
      )
    )
 

Fills `missings` by linear interpolation.
- `:dateinterval` => time period to use for grouping,

Example:

    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
    gval = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gmissing = 50000
    gndxmissing = Random.shuffle(1:length(gdate))[1:gmissing]
    X = DataFrame(Date=gdate,Value=gval)
    X.Value[gndxmissing] .= missing

    dnnr = DateValLinearImputer()
    fit!(dnnr,X)
    transform!(dnnr,X)

 
Implements: `fit!`, transform!`
"""
mutable struct DateValLinearImputer <: Transformer
  model
  args

  function DateValLinearImputer(args=Dict())
    default_args = Dict{Symbol,Any}(
        :dateinterval => Dates.Hour(1),
    )
    new(nothing,mergedict(default_args,args))
  end
end


"""
    fit!(dnnr::DateValLinearImputer,xx::T,y::Vector=[]) where {T<:DataFrame}

Validates and checks arguments for errors.
"""
function fit!(dnnr::DateValLinearImputer,xx::DataFrame,y::Vector=[])
  x = deepcopy(xx)
  validdateval!(x)
  dnnr.model=dnnr.args
end

"""
    transform!(dnnr::DateValLinearImputer,xx::T) where {T<:DataFrame}

Replaces `missings` by linear interpolation.
"""
function transform!(dnnr::DateValLinearImputer,xx::DataFrame)
  x = deepcopy(xx)
  validdateval!(x)
  valgator = DateValgator(dnnr.args)
  fit!(valgator,x)
  df=transform!(valgator,x)
  df.Value = interp(df.Value) |> locf() |> nocb()
  return df
end

end
