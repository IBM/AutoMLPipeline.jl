module Utils

export nested_dict_merge

"""
    nested_dict_merge(first::Dict, second::Dict)

Second nested dictionary is merged into first.

If a second dictionary's value as well as the first
are both dictionaries, then a merge is conducted between
the two inner dictionaries.
Otherwise the second's value overrides the first.

- `first`: first nested dictionary
- `second`: second nested dictionary

Returns: merged nested dictionary
"""
function nested_dict_merge(first::Dict, second::Dict)
   target = copy(first)
   for (second_key, second_value) in second
      values_both_dict =
      typeof(second_value) <: Dict &&
      typeof(get(target, second_key, nothing)) <: Dict
      if values_both_dict
         target[second_key] = nested_dict_merge(target[second_key], second_value)
      else
         target[second_key] = second_value
      end
   end
   return target
end

end
