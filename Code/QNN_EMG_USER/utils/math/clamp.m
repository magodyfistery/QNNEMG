function clamped_value = clamp(minimun, maximun, value)
% clamp a value between a range

if value < minimun
    clamped_value = minimun;
else
    if value > maximun
        clamped_value = maximun;
    else
        clamped_value = value;
    end
    
end

end

