def kalman_filter(self, error_est, prev_est, data):
        error_mea = data - prev_est
        KG = error_est/(error_est+error_mea)
        curr_est = prev_est + KG*(data-prev_est)
        new_error_est = (1-KG)*prev_est
        return new_error_est, curr_est