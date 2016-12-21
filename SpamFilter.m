
function main()
close all;
clear all;
fname = input('Enter a filename to load data for training/testing: ','s');
load(fname);

[feature_prob, class_prob]=NBTrain(AttributeSet, LabelSet);

[predicted_labels, accuracy]=NBTest(feature_prob, class_prob, testAttributeSet, validLabel);

[C, order]=confusionmat(validLabel, predicted_labels);

fprintf('********************************************** \n');
fprintf('Overall Accuracy on Dataset %s: %f \n', fname, accuracy);
fprintf('********************************************** \n');


end

function[training_data, class_prob]=NBTrain(AttributeSet, LabelSet)

feature_range=max(max(AttributeSet))+1;
class_range=max(LabelSet)+1;
training_data=zeros(class_range,(feature_range*57));

    for i=1:size(AttributeSet, 1)
        for j=1:57
            training_data(LabelSet(i,:)+1, (j-1)*feature_range+AttributeSet(i, j)+1)=training_data(LabelSet(i,:)+1, (j-1)*feature_range+AttributeSet(i, j)+1)+1;
        end
    end


    class_prob=zeros(class_range, 1);

    for i=1:class_range
        class_prob(i)=sum(training_data(i, 1:feature_range), 2);
%        training_data(i, :)=training_data(i, :)/class_prob(i);
    end

    total_test_values=sum(class_prob);

    for i=1:class_range
        for j=1:57
            if(ismember(0, training_data(i, (j-1)*feature_range+1:(j)*feature_range)))
                new_values=floor(total_test_values/100);
                for k=1:feature_range
                    training_data(i, (j-1)*feature_range+k)=(training_data(i, (j-1)*feature_range+k)+new_values*(1/feature_range))/(class_prob(i)+new_values);
                end
            else
                training_data(i, (j-1)*feature_range+1:(j)*feature_range)=training_data(i, (j-1)*feature_range+1:(j)*feature_range)/class_prob(i);
            end
        end
    end

    for i=1:class_range
        class_prob(i)=class_prob(i)/total_test_values;
    end
    

end

function[predictLabel, accuracy]=NBTest(feature_prob, class_prob, testAttributeSet, validLabel)

    feature_range=max(max(testAttributeSet))+1;
    class_range=max(validLabel)+1;
    predicted_class_prob=zeros(class_range, 1);
    predictLabel=zeros(size(testAttributeSet, 1), 1);
    correct=zeros(1);
    
    for k=1:size(testAttributeSet, 1)
        for i=1:class_range
            predicted_class_prob(i)=class_prob(i);
            for j=1:57
                predicted_class_prob(i)=predicted_class_prob(i)*(feature_prob(i, (j-1)*feature_range+testAttributeSet(k, j)+1));
            end
        end
    
        [x, I]=max(predicted_class_prob);
        predictLabel(k)=I-1;
        if(validLabel(k)==predictLabel(k))
            correct=correct+1;
        end
    end
    
    accuracy=correct/size(validLabel, 1);
end

