function [ Vertex_updated, Vertex_updatedR ] = SNS_optimization(Vertex_set_org, Vertex_warped_initial, importance_quad,importance_quadR, alpha_matrix, alpha_matrixR)
    % The Matlab codes are for non-comercial use only.
    [Vertex_updated, Vertex_updatedR ] = ...
        SNS_Opt_Iter_M3(Vertex_set_org ,Vertex_warped_initial, Vertex_warped_initial,importance_quad,importance_quadR, alpha_matrix, alpha_matrixR);

    Vertex_updated_old = Vertex_updated;

    Vertex_max_move = inf;
    Iter_NUM = 1;
    while(Vertex_max_move > 0.5)
        Iter_NUM = Iter_NUM + 1;
        disp(['########## Iteration no. ' num2str(Iter_NUM)]);
        [Vertex_updated, Vertex_updatedR] = ...
            SNS_Opt_Iter_M3(Vertex_set_org ,Vertex_updated, Vertex_updatedR, importance_quad,importance_quadR, alpha_matrix, alpha_matrixR);
        Vertex_max_move = max(max(max(abs(Vertex_updated_old - Vertex_updated))));
        Vertex_updated_old = Vertex_updated;
        disp(['--- Max movement =  ' num2str(Vertex_max_move, '%.3f')]);
    end

end

