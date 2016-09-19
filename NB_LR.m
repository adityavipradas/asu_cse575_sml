function [] = NB_LR(file)
    
    %CSE575 STATISTICAL MACHINE LEARNING
    %HOMEWORK 1
    %ADITYA VIPRADAS
    
    %clear screen
    clc;
    
    %read the csv file
    data = xlsread(file);
    disp(size(data));
    
    %data cleaning by removing NaN rows
    [row col] = find(isnan(data));
    data(row,:) = [];
    
    
    %remove 1st column
    data = data(:,2:size(data,2));
    disp(data)
    
    %random fractions
    ran_frac = [0.01 0.02 0.03 0.125 0.625 1];
    train_rows = 1:round((2/3)*size(data,1));
    
    %initialize
    pErr = [];
    finpErr = zeros(1,length(ran_frac))
    
    for times = 1:5
        for r = 1:length(ran_frac)
        
            %shuffle rows randomly
            train_rows = train_rows(randperm(length(train_rows)));
        
            %training and testing data extraction
    
            train = data(train_rows(1:round(ran_frac(r)*length(train_rows))),:);
            test = data(round((2/3)*size(data,1))+1:end, :);
    
            %***PERFORM GAUSSIAN NAIVE BAYES***
            X = train(:,1:size(train,2)-1);
            Y = train(:,size(train,2));
    
            %***LEARN THE DATA***
            %calculate mu and sigma2 for each feature
            [mu, sigma2] = calc_mu_sigma2(X,Y);
    
            %calculate the naive bayes probability for each test example
            P_malign = sum(Y == -1)/length(Y);
            P_benign = sum(Y == 1)/length(Y);
    
            error = 0
            for i = 1:size(test,1)
                Prob = naive_bayes(mu, sigma2, test(i,1:size(test,2)-1));
                Prob(1) = Prob(1)*P_malign;
                Prob(2) = Prob(2)*P_benign;
                normalize = sum(Prob);
                Prob = Prob/normalize
                [val, ind] = max(Prob)
                if ind==1 && test(i,size(test,2))==1
                    error = error + 1
                end
                if ind==2 && test(i,size(test,2))==-1
                    error = error + 1
                end
            end
            pErr(r) = error/size(test,1);
        end
        finpErr = finpErr + pErr;
    end
    finpErr = finpErr/5;
    disp(finpErr)
end

function [mu, sigma2] = calc_mu_sigma2(X,Y)
    %calculate mu and sigma2 for each feature
    row = 0;
    for k = -1:2:1 %binary classifier (-1 and 1)
        row = row + 1;
        for i = 1:size(X,2) %features
            den = 0;
            mu_num = 0;
            sig_num = 0;
            for j = 1:size(X,1) %training examples
                den = den + (Y(j,1) == k);
                mu_num = mu_num + X(j,i)*(Y(j,1) == k);
            end
            mu(row,i) = mu_num/den;
            for j = 1:size(X,1)
                sig_num = sig_num + (X(j,i) - mu(row,i))^2*(Y(j,1) == k);
            end
            sigma2(row,i) = sig_num/(den - 1);
        end
    end
end

function [Prob] = naive_bayes(mu, sigma2, testf)
    %conditional independence
    Prob = [1,1]; %[malign, benign]
    type = 0;
    for k = -1:2:1
        type = type + 1;
        for i = 1:length(testf)
            Prob(type) = Prob(type)* ...
                (1/sqrt(2*pi*sigma2(type,i)))* ...
                exp(-(testf(i)-mu(type,i))^2/(2*sigma2(type,i)));
        end
    end
end