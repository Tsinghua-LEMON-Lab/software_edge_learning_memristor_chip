classdef rram_array
    
    properties     
        weight_pos;
        weight_neg;
        weight_G;
        weight;
        weight_range;
        weight_G_scale;
        Gmin;
        Gmax;
        Gstep;
        row;
        col;
        b_RS;
        c_RS;
        pulse_num;
        inter_num;
        grad_bp;
        grad_bp_w_model;
        grad_sbp;
    end
    
    methods
        function a = rram_array(weight_pos, weight_neg, weight_G, weight, weight_range, weight_G_scale, b_RS, c_RS)
            
            a.weight_pos = weight_pos;
            a.weight_neg = weight_neg;
            a.weight_G = weight_G;
            a.weight = weight;
            a.weight_range = weight_range;
            a.weight_G_scale = weight_G_scale;
            a.Gmin = 2e-6;
            a.Gmax = 2e-5;
            a.Gstep = (a.Gmax - a.Gmin)/10;
            a.row = size(weight_pos, 1);
            a.col = size(weight_pos, 2);
            a.b_RS = b_RS;
            a.c_RS = c_RS;
            a.pulse_num = 0;
            a.inter_num = 0;
            a.grad_bp = [];
            a.grad_bp_w_model = [];
            a.grad_sbp = [];
        end
        
        function a = randomized_weight(a, rand_data_name)
           
            load(rand_data_name);
            a.weight = rand_data;                          
            a.weight = a.weight/max(a.weight(:))*a.weight_range/2.5;
            a.weight_G = a.weight/a.weight_G_scale;
            a.weight_pos = zeros(a.row, a.col) + (a.Gmax + a.Gmin)/2;  
            a.weight_neg =  a.weight_pos - a.weight_G;
            
            a.weight_pos(a.weight_pos < a.Gmin) = a.Gmin + 1e-9; 
            a.weight_neg(a.weight_neg > a.Gmax) = a.Gmax;  
        end
        
        function a = bp_update(a, y1, y2, label_onehot, lr, noise)
           
            
            error = label_onehot - y2;  
            grad_w2 = y1' * error; 
            weight_temp = a.weight;

%             grad_w2_flatten = grad_w2(:);
%             a.grad_bp = [a.grad_bp, grad_w2_flatten'];

            d_weight = lr * grad_w2;  
            d_weight_G = d_weight / a.weight_G_scale; 
            
            delta_w = d_weight_G/2; 
            a.weight_pos = a.weight_pos + delta_w;
            a.weight_neg = a.weight_neg - delta_w;
             
            if noise ~= 0
                noise_range = (a.Gmax - a.Gmin) * noise;
                a.weight_pos = a.weight_pos + (rand(a.row, a.col) - 0.5)*noise_range;
                a.weight_neg = a.weight_neg + (rand(a.row, a.col) - 0.5)*noise_range;
            end
            max(a.weight_pos, a.Gmin);  
            max(a.weight_neg, a.Gmin);
            min(a.weight_pos, a.Gmax);
            min(a.weight_neg, a.Gmax);
            
            a.weight_G = a.weight_pos - a.weight_neg;
            a.weight = a.weight_G * a.weight_G_scale;

        end
        
        function a = bp_update_th(a, y1, y2, label_onehot, lr, noise, threshold)

            error = label_onehot - y2; 

            error(abs(error) < threshold) = 0;
            grad_w2 = y1' * error; 
            weight_temp = a.weight;
%             grad_w2_flatten = grad_w2(:);
%             a.grad_bp = [a.grad_bp, grad_w2_flatten'];

            d_weight = lr * grad_w2; 
            d_weight_G = d_weight / a.weight_G_scale;  

            delta_w = d_weight_G/2;  
            a.weight_pos = a.weight_pos + delta_w;
            a.weight_neg = a.weight_neg - delta_w;

            if noise ~= 0
                noise_range = (a.Gmax - a.Gmin) * noise;
                a.weight_pos = a.weight_pos + (rand(a.row, a.col) - 0.5)*noise_range;
                a.weight_neg = a.weight_neg + (rand(a.row, a.col) - 0.5)*noise_range;
            end
            max(a.weight_pos, a.Gmin);   
            max(a.weight_neg, a.Gmin);
            min(a.weight_pos, a.Gmax);
            min(a.weight_neg, a.Gmax);
%             
            a.weight_G = a.weight_pos - a.weight_neg;
            a.weight = a.weight_G * a.weight_G_scale;
        end

        function a = circuit_sbp_update_single_update_wosz(a, y1, y2, label_onehot, threshold)
            
            sy = y1 > 0;  
            sz = y2 > 0;  
            error = label_onehot - y2;  
            error(abs(error) < threshold) = 0;
            se = sign(error);  
            dw2 = sy' * (se);
            grad = -dw2; 

%             grad_w2_flatten = grad(:);
%             a.grad_sbp = [a.grad_sbp, grad_w2_flatten'];

            weight_temp = a.weight;
            grad_sign = sign(grad);  
           
            if a.inter_num == 0   
                weight_pos_op_row = -grad_sign + 2; 
                weight_pos_op_row(weight_pos_op_row==3) = 2;  
                weight_neg_op_row = grad_sign + 2;
                weight_neg_op_row(weight_neg_op_row==3) = 2; 

                
    %             a.weight_pos = model_fit(a.weight_pos, weight_pos_op_row, a.b_RS, a.c_RS);
    %             a.weight_neg = model_fit(a.weight_neg, weight_neg_op_row, a.b_RS, a.c_RS);
                a.weight_pos = a.model_fit1(a.weight_pos, weight_pos_op_row);
                a.weight_neg = a.model_fit1(a.weight_neg, weight_neg_op_row);
                a.weight_G = a.weight_pos - a.weight_neg;
                a.weight = a.weight_G * a.weight_G_scale;
          
                a.pulse_num = a.pulse_num + sum(abs(grad_sign(:)));  
                a.inter_num = 1;
            elseif a.inter_num == 1  

                weight_pos_op_row = -grad_sign + 2;  
                weight_pos_op_row(weight_pos_op_row==1) = 2;  
                weight_neg_op_row = grad_sign + 2;
                weight_neg_op_row(weight_neg_op_row==1) = 2;  
 
    %             a.weight_pos = model_fit(a.weight_pos, weight_pos_op_row, a.b_RS, a.c_RS);
    %             a.weight_neg = model_fit(a.weight_neg, weight_neg_op_row, a.b_RS, a.c_RS);
                a.weight_pos = a.model_fit1(a.weight_pos, weight_pos_op_row);
                a.weight_neg = a.model_fit1(a.weight_neg, weight_neg_op_row);
                a.weight_G = a.weight_pos - a.weight_neg;
                a.weight = a.weight_G * a.weight_G_scale;

                a.pulse_num = a.pulse_num + sum(abs(grad_sign(:)));  
                a.inter_num = 0;
            end
        end
        
        function Gnext = model_fit1(a, G, op_row)
           
            n_range = ceil((G - a.Gmin)/a.Gstep); 
            b_op = a.b_RS(op_row+(n_range-1)*3); 
            c_op = a.c_RS(op_row+(n_range-1)*3);
            c_op_t = c_op/sqrt(2);  

            deltaG = c_op_t .* randn(a.row, a.col) + b_op;
            sigma_pp = c_op_t + b_op;  
            sigma_pn = -c_op_t + b_op;
            deltaG(deltaG > sigma_pp) = sigma_pp(deltaG > sigma_pp);
            deltaG(deltaG < sigma_pn) = sigma_pn(deltaG < sigma_pn);
            Gnext = deltaG + G;
            Gnext(Gnext < a.Gmin) = a.Gmin + 1e-9;  
            Gnext(Gnext > a.Gmax) = a.Gmax; 
        end
        
        function a = bp_update_pulse(a, y1, y2, label_onehot, lr, noise, pulse_bound)
            
            error = label_onehot - y2; 
            grad_w2 = y1' * error;  

            %******************这里是我们用来保存grad的语句
%             grad_w2_flatten = grad_w2(:);
%             a.grad_bp_w_model = [a.grad_bp_w_model, grad_w2_flatten'];

            d_weight = lr * grad_w2;  
            d_weight_G = d_weight / a.weight_G_scale; 

           
            delta_w = d_weight_G/2;  
            weight_pos_target = a.weight_pos + delta_w; 
            weight_neg_target = a.weight_neg - delta_w;
            max(weight_pos_target, a.Gmin); 
            max(weight_neg_target, a.Gmin);
            min(weight_pos_target, a.Gmax);
            min(weight_neg_target, a.Gmax);
            tol = (a.Gmax - a.Gmin) * noise / 2;  
            
            p_num = 0;
            while(p_num < pulse_bound)  
               
                weight_pos_deviation = weight_pos_target - a.weight_pos;  
                weight_neg_deviation = weight_neg_target - a.weight_neg;
                
                weight_pos_deviation(abs(weight_pos_deviation) < tol) = 0;  
                weight_neg_deviation(abs(weight_neg_deviation) < tol) = 0;

                if any(weight_pos_deviation(:)) 
                    weight_pos_op = sign(weight_pos_deviation);  
                    weight_pos_op_row = weight_pos_op + 2; 
                    
                    a.weight_pos = a.model_fit1(a.weight_pos, weight_pos_op_row);
                    a.pulse_num = a.pulse_num + sum(abs(weight_pos_op(:)));
                end
                
                if any(weight_neg_deviation(:))  
                    weight_neg_op = sign(weight_neg_deviation); 
                    weight_neg_op_row = weight_neg_op + 2;  
                    
                    a.weight_neg = a.model_fit1(a.weight_neg, weight_neg_op_row);
                    a.pulse_num = a.pulse_num + sum(abs(weight_neg_op(:)));
                end
                
                p_num = p_num + 1;
            end

            a.weight_G = a.weight_pos - a.weight_neg;
            a.weight = a.weight_G * a.weight_G_scale;   
        end

        function a = bp_update_pulse_wo_verify(a, y1, y2, label_onehot, lr, noise)
           
            pulse_bound = 1;
           
            error = label_onehot - y2;  
            grad_w2 = y1' * error; 
            d_weight = lr * grad_w2; 
            d_weight_G = d_weight / a.weight_G_scale; 

           
            delta_w = d_weight_G/2; 
            weight_pos_target = a.weight_pos + delta_w; 
            weight_neg_target = a.weight_neg - delta_w;
            max(weight_pos_target, a.Gmin); 
            max(weight_neg_target, a.Gmin);
            min(weight_pos_target, a.Gmax);
            min(weight_neg_target, a.Gmax);
            tol = (a.Gmax - a.Gmin) * noise / 2; 
            
            p_num = 0;
            while(p_num < pulse_bound)  
                
                weight_pos_deviation = weight_pos_target - a.weight_pos;  
                weight_neg_deviation = weight_neg_target - a.weight_neg;
                
                weight_pos_deviation(abs(weight_pos_deviation) < tol) = 0; 
                weight_neg_deviation(abs(weight_neg_deviation) < tol) = 0;

                if any(weight_pos_deviation(:))  
                    weight_pos_op = sign(weight_pos_deviation); 
                    weight_pos_op_row = weight_pos_op + 2;  
                    
                    a.weight_pos = a.model_fit1(a.weight_pos, weight_pos_op_row);
                    a.pulse_num = a.pulse_num + sum(abs(weight_pos_op(:)));
                end
                
                if any(weight_neg_deviation(:))  
                    weight_neg_op = sign(weight_neg_deviation);  
                    weight_neg_op_row = weight_neg_op + 2;  
                    
                    a.weight_neg = a.model_fit1(a.weight_neg, weight_neg_op_row);
                    a.pulse_num = a.pulse_num + sum(abs(weight_neg_op(:)));
                end
                
                p_num = p_num + 1;
            end

            a.weight_G = a.weight_pos - a.weight_neg;
            a.weight = a.weight_G * a.weight_G_scale;   
        end
        
    end
    
end