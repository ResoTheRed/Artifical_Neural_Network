% T. Hoque: Here we are solving our popular Cancer{Benign, Malignant} toy problem using ANN {Back_Prop based}
% Layer_node information: L=[2 2 2 2], you can chance to see the effect.


% ///////////////////////////////////////////////// Training Section ////////////////////////////////////////////////
% ////////// initialization

L=[4 3 3];   % // Defining the layers: Total of 4 layers, # of nodes are 2, 2, 2, 2 respectively from input to output layer
alpha = 0.2;   % //usually alpha > 0, ranging from 0.1 to 1
target_mse=0.0001 % // one of the exit condition
Max_Epoch=2000   % // one of the exit condition
Min_Error=Inf
Min_Error_Epoch=-1
epoch=0;       % // 1 epoch => One forward and backward sweep of the net for each training sample 
mse =Inf;      % // initializing the Mean Squared Error with a very large value.
Err=[];
Epo=[];
acc = 0;
acc_max = 0;

% ////////// load datasets
X = importdata('train_X.txt'); % // contains features: Column1: x1 (size) and Column2: x2 (age)
Y = importdata('train_Y.txt'); % // Target Output 
[Nx,P]=size(X); % // Nx = # of sample in X, P= # of feature in X
[Ny,K]=size(Y); % // Ny = # of target output in Y, K= # of class for K classes when K>=3 otherwise, K=2 in this binary case now

% Optional: Since input and output are kept in different files, it is better to verify the loaded sample size/dimensions.
if Nx ~= Ny 
      error ('The input/output sample sizes do not match');
end


% Optional
if L(1) ~= P
       error ('The number of input nodes must be equal to the size of the features')' 
end 

% Optional
if L(end) ~= K
       error ('The number of output node should be equal to K')' 
end 


B=cell(length(L)-1,1);  % forming the number of Beta/weight matrix needed in between the layers

for i=1:length(L)-1        % Assign uniform random values in [-0.7, 0.7] 
      B{i} =[1.4.*rand(L(i)+1,L(i+1))-0.7];	
end 

%Let us allocate places for Term, T 
T=cell(length(L),1);
for i=1:length(L)
	T{i} =ones (L(i),1);
end


%Let us allocate places for activation, i.e., Z
Z=cell(length(L),1);
for i=1:length(L)-1
	Z{i} =zeros (L(i)+1,1); % it does not matter how do we initialize (with '0' or '1', or whatever,) this is fine!
end
Z{end} =zeros (L(end),1);  % at the final layer there is no Bias unit

%Let us allocate places for error term delta, d
d=cell(length(L),1);
for i=1:length(L)
	d{i} =zeros (L(i),1);
end
			

Z{1} = [X ones(Nx, 1)]';  
Y=Y';   	   
  

while ((mse > target_mse) && (epoch < Max_Epoch))   
 
  CSqErr=0; 		
      % // forward propagation 
      for i=1:length(L)-1
       	     T{i+1} = B{i}' * Z{i};
            
             if (i+1)<length(L)
               Z{i+1}=[(1./(1+exp(-T{i+1}))); ones(Nx,1)'];
             else  
               Z{i+1}=(1./(1+exp(-T{i+1}))); 
             end 
       end  

         
      CSqErr= CSqErr+sum(sum(((Y-Z{end}).^2),1));   
      CSqErr = CSqErr/L(end);  % Normalizing the Error based on the number of output Nodes

     % // Compute error term delta 'd' for each of the node except the input unit
     % // -----------------------------------------------------------------------
         d{end}=(Z{end}-Y).*Z{end}.*(1-Z{end}); % // delta error term for the output layer.

         for i=length(L)-1:-1:2 
               W=Z{i}(1:end-1,:).*(1-Z{i}(1:end-1,:)); D= d{i+1}';
               for m = 1:Nx                             
                   d{i}(:,m)=W(:,m).*sum((D(m,:).*B{i}(1:end-1,:)),2);  % // sum(A, 2) => row wise sum of matrix A
               end
         end              
       
      % // Now we will update the parameters/weights
      for i=1:length(L)-1 
                W = Z{i}(1:end-1,:);   V1 = zeros(L(i),L(i+1));   V2 = zeros(1,L(i+1));    D = d{i+1}';
           for m = 1:Nx
                V1 = V1 + (W(:,m)*D(m,:));   V2 = V2 + D(m,:);     
           end 
          
           B{i}(1:end-1,:)=B{i}(1:end-1,:)-(alpha/Nx).*V1;             
           B{i}(end,:) = B{i}(end,:)-(alpha/Nx).*V2;  			
       end 
     
  
    CSqErr= (CSqErr) /(Nx);        % //Average error of N sample after an epoch 
    mse=CSqErr;
    epoch  = epoch+1;
    
    Err = [Err mse];
    Epo = [Epo epoch];   


    if mse < Min_Error
	    Min_Error=mse;
      Min_Error_Epoch=epoch;
    end 
					    	
 end % //while_end

      Min_Error
      Min_Error_Epoch 

%//=============================================================================================================================
%//The NN Node and structure needs to be saved, i.e. save L
    L
	
%// Now the predicted weight B with least error should be saved in a file to be loaded and to be used for test set/new prediction

  for i=1:max(size(B))
     B{i}
  end 
  
%plot epoch versus error graph
hold on;
plot (Epo,Err)  % plot based on full epoch
plot (Epo(1:1000),Err(1:1000))
plot (Epo(1:400),Err(1:400))  
%title ("FCV 1 Training Gamma=0.33");


%// ================================================================================================================
% ///////////////////////////////////////////////// Test Section ////////////////////////////////////////////////
% // Here I will be using the last B computed to demo test data to classify but you should save and use best B. 
% // Feed forward part will actually be used, assume test points: 1.(0.5,0.3) and 2.(5,4)
% // NOTE: For point (1) the output is expected to be close to zero
% //      For point (2) the output is expected to be close to one.


 X = importdata('test_X.txt');
 Y = importdata('test_Y.txt');
 [Ntest, q] = size(X);
 TestErr=0;
 testErrArr=[];
 acc = 0;

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
       [max_x, index_x] = max(Z{end});
       [max_y, index_y] = max(Yk);

       if index_x == index_y
        acc = acc+1;
       end

       TestErr = TestErr+sum(sum(((Yk-Z{end}).^2),1));   
   end 
 
 acc = acc/Ntest

TestErr= (TestErr) /(Ntest);   % //Average error of N sample after an epoch 
CSqErr = CSqErr/L(end);  % Normalizing the Error based on the number of output Nodes
mse=TestErr; 
testErrArr = [testErrArr mse];

Err = [Err mse];
Epo = [Epo epoch]; 

%===============================================================================================================
acc
mse

plot (Epo,testErrArr)  % plot based on full epoch
plot (Epo(1:1000),Err(1:1000))
plot (Epo(1:400),Err(1:400))  
hold off;
  
