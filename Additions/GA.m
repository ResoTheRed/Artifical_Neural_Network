clear;

clc;

L = [4 3 3 3 3];
[l_row, l_Col] = size(L);
pop_size = 10;

mutation_rate = pop_size * 0.3;
crossover_rate = pop_size* 0.8;
elite_rate = pop_size * 0.1;
gen_limit = 10000;

% //GenerateDataset
% //Initialize Population

function [beta] = initializePopulation(pop_size, L)

  beta=[];
  for i=1:pop_size
    beta=[beta getBeta(L)];
  end

end  
% //Assign uniform Beta random values in
function [B] = getBeta(L)

  B=cell(length(L)-1,1); 
  for i=1:length(L)-1 
    B{i} =-1000+(2000).*rand(L(i)+1,L(i+1));
  end

end 
% //Do the cross over

function [beta] = crossOverPopulation(pop_size, beta, crossover_rate, col)
  count = 0; 
  while count < crossover_rate
    randRow = randi([1 col-1]); %generate random num to select the row for crossover
    randCol = randi([1 pop_size-1]); %generate the random num to select the column. pop_size-1 because, I will take next row for swapping. If the random num = pop_size there will be an error
    %//fprintf('%d X %d swap with %d X %d\n', randRow, randCol, randRow, randCol+1);
    mat1 = beta{randRow, randCol};
    mat2 = beta{randRow, randCol+1};
    beta{randRow, randCol} = mat2;
    beta{randRow, randCol+1} = mat1;
    count = count+2;

  end

end
%Do mutation

function[beta] = mutatePopulation(pop_size, beta, mutation_rate, col)

  count = 0;

  while count < mutation_rate

    randRow = randi([1 col-1]); %generate random num to select the row for crossover
    randCol = randi([1 pop_size]);
    %fprintf('Mutate %d X %d \n', randRow, randCol);
    randomBetaSelection = beta{randRow, randCol}
    [size_rand_row size_rand_col] = size(randomBetaSelection);
    randRowSelection = randi([1 size_rand_row]);

    for i=1:size_rand_col
      fprintf('%d X %d -> (%d %d) %d \n', randRow, randCol, randRowSelection, i, beta{randRow, randCol}(randRowSelection,i));
      beta{randRow, randCol}(randRowSelection,i) = randi([-1000,1000])*rand(1,1); % assign random value to those cols
    end

    beta{randRow, randCol};  
    count = count + 1;

  end  

end
beta = initializePopulation(pop_size, L);
crossedPop = crossOverPopulation(pop_size, beta, crossover_rate, l_Col);
mutatedPopulation = mutatePopulation(pop_size, crossedPop, mutation_rate, l_Col);

X = importdata('train_X.txt');
Y = importdata('train_Y.txt');
counter = 1;
while counter < pop_size
    B = beta{counter};
    counter = counter+1;
    [Ntest, q] = size(X);
    TestErr=0;
    testErrArr=[];

    %// ====== Same (or similar) code as we used before for feed-forward part (see above)
    for j=1: Ntest		    % for loop #1		
        Z{1} = [X(j,:) 1]';   % Load Inputs with bias=1
        Yk   = Y(j,:)'; 	  % Load Corresponding Desired or Target output
        % // forward propagation 
        % //----------------------
        for i=1:length(L)-1
                T{i+1} = B{i}' * Z{i};
                if (i+1)<length(L)
                Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
                else  
                Z{i+1}=(1./(1+exp(-T{i+1}))); 
                end 
        end  % //end of forward propagation 
        Z{end};
        TestErr= TestErr+sum(sum(((Yk-Z{end}).^2),1));   
    end 
    

    TestErr= (TestErr) /(Ntest);   % //Average error of N sample after an epoch 
    mse=TestErr  
        
    %===============================================================================================================
    mse
end

L = [4,3,3,3,3]

B=cell(length(L)-1,1);
B{1} =    [
        [0.397263,-0.440996,0.224320],
        [0.628144,0.598893,-0.628478],
        [0.595008,-0.348609,-0.208487],
        [-0.074460,0.599416,0.621480],
        [0.206494,0.347336,-0.578062]
    ]
B{2} =    [
        [-0.214108,-0.251251,0.595624],
        [0.195364,-0.583371,-0.245872],
        [-0.121688,0.511927,-0.635016],
        [0.028181,-0.622591,0.689460]
    ]
B{3} =    [
        [-0.4656388,  -0.4580941,   0.0034051],
        [-0.0534615,   0.3845925,  -0.2333062],
        [0.5270184,  -0.2597160,  -0.1325904],
        [0.6857013,   0.6930017,   0.4828105]
    ]
B{4} =    [
        [0.175076,   0.317678,  -0.669430],
        [-0.599514,  -0.526939,   0.396396],
        [0.440614,   0.166052,   0.684786],
        [-0.053484,   0.644765,  -0.332944]
    ]

 X = importdata('test_X.txt');
 Y = importdata('test_Y.txt');
 [Ntest, q] = size(X);
 TestErr=0;
 testErrArr=[];

%// ====== Same (or similar) code as we used before for feed-forward part (see above)
  for j=1: Ntest		    % for loop #1		
      Z{1} = [X(j,:) 1]';   % Load Inputs with bias=1
      Yk   = Y(j,:)'; 	  % Load Corresponding Desired or Target output
      % // forward propagation 
      % //----------------------
      for i=1:length(L)-1
       	     T{i+1} = B{i}' * Z{i};
             if (i+1)<length(L)
               Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
             else  
               Z{i+1}=(1./(1+exp(-T{i+1}))); 
             end 
      end  % //end of forward propagation 
       Z{end};
       %TestErr
       (Yk-Z{end})
       TestErr= TestErr+sum(sum(((Yk-Z{end}).^2),1));   
   end 
 

TestErr= (TestErr) /(Ntest);   % //Average error of N sample after an epoch 
TestErr = TestErr / L(end);
mse=TestErr; 
    
%===============================================================================================================
mse