function make(command)

% description: make file for all the C and MEX code
% author: Ribana Roscher (rroscher@uni-bonn.de)
% date: August 2012 (last modified)

disp('Compile mex files.')

if (nargin > 0 && strcmp(command,'clean'))
    delete('*.mexglx');
    delete('*.mexa64');
    delete('*.mexw32');
    delete('*.mexw64');
    return;
end

mex CC=g++ compute_kernel.cpp -Ieigen
mex CC=g++ test_importPoints.cpp -Ieigen
mex CC=g++ test_importPoints_fly.cpp -Ieigen

disp('Done.')
