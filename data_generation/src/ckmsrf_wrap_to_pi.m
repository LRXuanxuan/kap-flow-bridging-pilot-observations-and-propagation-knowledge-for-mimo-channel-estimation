function y = ckmsrf_wrap_to_pi(x)
%CKMSRF_WRAP_TO_PI Wrap angles to [-pi, pi).
y = mod(x + pi, 2*pi) - pi;
end
