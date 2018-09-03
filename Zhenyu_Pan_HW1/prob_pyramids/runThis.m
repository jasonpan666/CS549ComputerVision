
clear all; 
close all;
clc;

im = rgb2gray(imread('data/bug.jpg'));
N=5;
G={};
L={};
[G,L]=pyramidsGL(im,N);
displayPyramids(G,L);
displayFFT(G,L);