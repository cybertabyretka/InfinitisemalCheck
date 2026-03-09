pub struct Infinitesimal {
    func: Box<dyn Fn(f64) -> f64>,

    pub slope: Option<f64>,
    pub intercept: Option<f64>,   
}

impl Infinitesimal {
    pub fn new<F>(func: F) -> Self
    where
        F: Fn(f64) -> f64 + 'static,
    {
        Infinitesimal {
            func: Box::new(func),

            slope: None,
            intercept: None,
        }
    }

    pub fn is_infinitesimal(&self, x_start: f64, x_end: f64, n: usize, tol: f64, dec_tol: f64) -> Result<bool, String> {
        if !(x_start > x_end && x_end > 0.0) {
            return Err("x_start must be greater than x_end and both must be positive".into());
        }
        let xs = geometric_sequence(x_start, x_end, n);
        let ys: Vec<f64> = xs.iter().map(|&x| (self.func)(x)).collect();
        let abs_vals: Vec<f64> = ys.iter().map(|y| y.abs()).collect();
        if let Some(&last_val) = abs_vals.last() {
            if last_val > tol {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }
        let mut decreases = 0usize;
        for i in 0..abs_vals.len().saturating_sub(1) {
            if abs_vals[i + 1] <= abs_vals[i] {
                decreases += 1;
            }
        }
        Ok(decreases >= (abs_vals.len().saturating_sub(1) as f64 * dec_tol) as usize)
    }

    pub fn build_tables(&mut self, x_start: f64, x_end: f64, n: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
        if !(x_start > x_end && x_end > 0.0) {
            return Err("x_start must be greater than x_end and both must be positive".into());
        }
        let x = geometric_sequence(x_start, x_end, n);
        let y: Vec<f64> = x.iter().map(|&x| (self.func)(x)).collect();
        let mut lxs: Vec<f64> = Vec::new();
        let mut lys: Vec<f64> = Vec::new();
        let mut lxs_clean: Vec<f64> = Vec::new();
        let mut lys_clean: Vec<f64> = Vec::new();
        for (&x, &y) in x.iter().zip(y.iter()) {
            let lx = if x > 0.0 { x.log10() } else { f64::NAN };
            let ly = y.abs().log10();
            lxs.push(lx);
            lys.push(ly);
            if lx.is_finite() && ly.is_finite() {
                lxs_clean.push(lx);
                lys_clean.push(ly);
            }
        }
        if lxs_clean.len() < 2 {
            return Err("Not enough valid points".into());
        }
        Ok((lxs, lys, lxs_clean, lys_clean))
    }

    pub fn compute_log_log_approximation(&mut self, lxs_clean: Vec<f64>, lys_clean: Vec<f64>) -> Result<(), String> {
        if lxs_clean.len() < 2 {
            return Err("Not enough valid points for regression".into());
        }
        let n = lxs_clean.len();
        let x_mean = lxs_clean.iter().sum::<f64>() / n as f64;
        let y_mean = lys_clean.iter().sum::<f64>() / n as f64;
        let mut xy_cov = 0f64;
        let mut x_var = 0f64;
        for i in 0..n {
            xy_cov += (lxs_clean[i] - x_mean) * (lys_clean[i] - y_mean);
            x_var += (lxs_clean[i] - x_mean).powi(2);
        }
        if x_var == 0.0 {
            return Err("Variance of log(x) is zero".into());
        }
        let slope = xy_cov / x_var;
        let intercept = y_mean - slope * x_mean;
        self.slope = Some(slope);
        self.intercept = Some(intercept);
        Ok(())
    }

    pub fn get_approximation_params(&mut self) -> Result<(f64, f64), String> {
        if self.slope.is_none() || self.intercept.is_none() {
            return Err("Approximation parameters not computed, please run compute_log_log_approximation first".into());
        }
        let alpha = self.slope.ok_or("slope is None")?;
        let c = 10f64.powf(self.intercept.ok_or("intercept is None")?);
        Ok((alpha, c))
    }
}


fn geometric_sequence(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![a];
    }
    let mut sequence = Vec::with_capacity(n);
    let ratio = (b / a).powf(1.0 / (n - 1) as f64);
    for i in 0..n {
        sequence.push(a * ratio.powf(i as f64));
    }
    sequence
}


fn main() {
    let x_start = 1e-1;
    let x_end = 1e-10;
    let n = 100usize;
    let tol = 1e-6;
    let dec_tol = 0.8;
    let mut inf = Infinitesimal::new(|x| 3.0 * x.powf(1.5));
    let is_infsml = inf.is_infinitesimal(x_start, x_end, n, tol, dec_tol).unwrap_or_else(|e| {
            println!("Error checking infinitesimal: {}", e);
            false
        });
    println!("f(x) = 3 * x^1.5 is infinitesimal as x -> 0: {}", is_infsml);
    let (_, _, lxs_clean, lys_clean) = inf.build_tables(x_start, x_end, n).unwrap_or_else(|e| {
        println!("Error building tables: {}", e);
        (Vec::new(), Vec::new(), Vec::new(), Vec::new())
    });
    match inf.compute_log_log_approximation(lxs_clean, lys_clean) {
        Ok(()) => {
            let (alpha, c) = inf.get_approximation_params().unwrap();
            println!("log-log regression: slope(alpha) = {:.6}, lg(C) = {:.6}", alpha, c.log10());
            println!("estimated: alpha = {:.6}, C = {:.6}", alpha, c);
            println!("expected: alpha=1.5, C=3");
        }
        Err(e) => println!("Regression error: {}", e),
    }
}