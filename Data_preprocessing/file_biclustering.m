% read file
M = csvread('binary_values.csv');

% a modifier for bicluster file
% the first number means how many users, the second number means how many items
% the last number means the maximum number of biclusters
biclusters = bimax(M, 5, 7, 1000);
% save data in the file
filename = 'biclusters.csv';
fid = fopen(filename, 'a+');

% struct file: filed + (rows & cols)
for i =1: biclusters.ClusterNo
    
    a = biclusters.Clust(i).rows;
    b = biclusters.Clust(i).cols;

    % rows can be output directly
    fprintf(fid, '%d ', a);
    fprintf(fid, '\n');
    
    % cols needs to be output with iteration
    for k =1: size(b)
        fprintf(fid, '%d ', b);
    end
     fprintf(fid, '\n');
end

fclose(fid);