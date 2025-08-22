function combine()
  a = hcat(caretdriver(), skaddriver())
  y = hcat(a, mean.(eachrow(a)))
  #@show y
  #y[findall(x -> x > 0.3, y.x1), :]
end
combine()

