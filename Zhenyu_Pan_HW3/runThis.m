% Zhenyu Pan HW3 CS/RBE549 Computer Vision WPI
% Please run this first and results will be generated and saved
% Two functions will be used

errors1 = eigenfaces(9, ones(1));
errors2 = eigenfaces(9, [1, 5]);
errors3 = eigenfaces(30, ones(1));
errors4 = eigenfaces(30, [1, 5]);

errors5 = fisherfaces(10, ones(1));
errors6 = fisherfaces(10, [1, 5]);
errors7 = fisherfaces(31, ones(1));
errors8 = fisherfaces(31, [1, 5]); 