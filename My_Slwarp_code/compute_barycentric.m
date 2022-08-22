function alphas = compute_barycentric(p,v1,v2,v3,v4)
    % Data values
    %p=[2 2];
    %v1=[1 1];
    %v2=[3 2];
    %v3=[2 4];
    %v4=[0 3];
    % define the previous notation
    a=(v1-p);
    b=v2-v1;
    c=v4-v1;
    d=v1-v2-v4+v3;

    % compute 2on order equation A mu^2 + B mu + C=0
    % as the vertices are 2D, we add a zero third component
    % to compute cross products.
    A=cross([c,0],[d,0]); %must be 3D vectors
    B=cross([c,0],[b,0])+cross([a,0],[d,0]);
    C=cross([a,0],[b,0]);
    % Only third component is needed (the other two are zero)
     A=A(3);
     B=B(3);
     C=C(3);
    %
    % Check for unique solutions
    %
    if (abs(A)<1.e-14)
        u1= -C/B;
        u2=u1;
    else
        %
        % Check for non complex solutions
        %
        if (B^2-4*A*C >0)
            u1=(-B+sqrt(B^2-4*A*C))/(2*A);
            u2=(-B-sqrt(B^2-4*A*C))/(2*A);
        else %complex solution
            u1=-1000;
            u2=u1;
        end
    end
    %
    mu=-10000; %stupid value small enough
    if(u1>=0 && u1<=1)
        mu=u1;
    end
    if(u2>=0 && u2<=1)
        mu=u2;
    end

    % compute 2on order equation A lambda^2 + B lambda + C=0
    A=cross([b,0],[d,0]); %must be 3D vectors
    B=cross([b,0],[c,0])+cross([a,0],[d,0]);
    C=cross([a,0],[c,0]);
    % Only third component is needed (the other two are zero)
    A=A(3);
    B=B(3);
    C=C(3);
    %
    % Check for unique solutions
    %
    if (abs(A)<1.e-14)
        w1= -C/B;
        w2=w1;
    else
        %
        % Check for non complex solutions
        %
        if (B^2-4*A*C >0)
            w1=(-B+sqrt(B^2-4*A*C))/(2*A);
            w2=(-B-sqrt(B^2-4*A*C))/(2*A);
        else %complex solution
            w1=-1000;
            w2=w1;
        end
    end
    %
    lambda=-10000; %stupid value
    if(w1>=0 && w1<=1)
        lambda=w1;
    end
    if(w2>=0 && w2<=1)
        lambda=w2;
    end
    %[mu,lambda] %parameters
    % Barycentric coordinates
    alpha1=(1-mu)*(1-lambda);
    alpha2=lambda*(1-mu);
    alpha3=mu*lambda;
    alpha4=(1-lambda)*mu;
    alphas=[alpha1,alpha2,alpha3,alpha4];
    % obtained point
    %q=alpha1*v1+alpha2*v2+alpha3*v3+alpha4*v4;
    %p-q %we must recover the same point
    % Plot the results
    %vertices=[v1,v2,v3,v4];
    %plotRectangle(v1,v2,v3,v4);
    %%%plot the result
    %figure;
    %line([v1(1) v2(1)],[v1(2) v2(2)]);
    %line([v2(1) v3(1)],[v2(2) v3(2)]);
    %line([v3(1) v4(1)],[v3(2) v4(2)]);
    %line([v4(1) v1(1)],[v4(2) v1(2)]);
    %hold on;
    %plot(p(:,1),p(:,2),'o');
    %plot(q(:,1),q(:,2),'*');
    %hold off;
end