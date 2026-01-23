package main

import "core:encoding/json"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os/os2"

Arm :: struct
{
    true_value, estimated_value: f64,
    times_taken: u32,
    preference_value: f64,
}

Figure :: struct
{
    plots: []Plot,
    x_label, y_label: string,
    x_max: f64,
    title: string,
}

Plot :: struct
{
    label: string,
    x_data, y_data: []f64,
}

Output :: struct
{
    figures: []Figure
}

main :: proc() {
    ARMS :: 10
    STEPS :: 200_000
    PERFORMANCE_MEASUREMENT_START_STEPS :: 100_000
    RUNS :: 1000

    EPSILON_VALUES :: []f64 { 1/128.0, 1/64.0, 1/32.0, 1/16.0, 1/8.0, 1/4.0}
    epsilon_greedy_reward_average := make([]f64, len(EPSILON_VALUES))
    
    for epsilon, i in EPSILON_VALUES
    {
        ALPHA :: 0.1
        for j in 1 ..= RUNS
        {
            run_average_reward: f64

            // select starting action values
            true_value := rand.float64_normal(mean = 0, stddev = 1)
            arms: [ARMS]Arm
            for &arm in arms
            {
                arm.true_value = true_value
            }

            for k in 0 ..< STEPS
            {
                // update arm every step to simulate a nonstationary problem
                for &arm in arms
                {
                    arm.true_value += rand.float64_normal(mean = 0, stddev = 0.01)
                }
            
                should_explore := rand.float64() < epsilon

                chosen_arm_index: int
                if should_explore
                {
                    chosen_arm_index = rand.int_max(ARMS)
                } else
                {
                    greedy_options := make([dynamic]int, context.temp_allocator)
                    best_value := min(f64)
                    for arm, arm_index in arms
                    {
                        if arm.estimated_value >= best_value
                        {
                            if arm.estimated_value > best_value
                            {
                                clear(&greedy_options)
                                best_value = arm.estimated_value
                            }
                            append(&greedy_options, arm_index)
                        }
                    }
                    chosen_arm_index = rand.choice(greedy_options[:])
                }
                chosen_arm := &arms[chosen_arm_index]
            
                chosen_arm_reward := rand.float64_normal(mean = chosen_arm.true_value, stddev = 1)

                if steps_since_measurement_start := k - PERFORMANCE_MEASUREMENT_START_STEPS; steps_since_measurement_start > 0
                {                   
                    run_average_reward += (chosen_arm_reward - run_average_reward) / f64(steps_since_measurement_start)
                }
                
                chosen_arm.estimated_value += ALPHA * (chosen_arm_reward - chosen_arm.estimated_value)
            }

            free_all(context.temp_allocator)
            epsilon_greedy_reward_average[i] += (run_average_reward - epsilon_greedy_reward_average[i]) / f64(j)
        }
    }

    
    ALPHA_VALUES :: []f64 { 1/32.0, 1/16.0, 1/8.0, 1/4.0, 1/2.0, 1, 2}
    gradient_reward_average := make([]f64, len(ALPHA_VALUES))
    foo := true
    for alpha, i in ALPHA_VALUES
    {
        for j in 1 ..= RUNS
        {
            run_average_reward: f64

            // select starting action values
            true_value := rand.float64_normal(mean = 0, stddev = 1)
            arms: [ARMS]Arm
            for &arm in arms
            {
                arm.true_value = true_value
            }
            rewards_average: f64

            for k in 1 ..= STEPS
            {
                // update arm every step to simulate a nonstationary problem
                for &arm in arms
                {
                    arm.true_value += rand.float64_normal(mean = 0, stddev = 0.01)
                }
                
                arm_natural_exponential_cache: [ARMS]f64
                // 2.11 page 37 Reinforcement Learning, An Introduction, Second Edtion by Sutton and Barto
                // Sum of e to the power of each preference value
                preference_natural_exponential_sum: f64
                for arm, i in arms
                {
                    pref_natural_exponential := math.pow(math.E, arm.preference_value)
                                        
                    arm_natural_exponential_cache[i] = pref_natural_exponential
                    preference_natural_exponential_sum += pref_natural_exponential
                }

                arm_probabilities: [ARMS]f64
                for arm, i in arms
                {
                    pref_natural_exponential := arm_natural_exponential_cache[i] 
                    arm_probability := pref_natural_exponential / preference_natural_exponential_sum

                    previous_probability: f64
                    arm_probabilities[i] = arm_probability
                }
        
                chosen_arm_index: int

                rand_float := rand.float64()

                previous_probability: f64
                for arm_prob, i in arm_probabilities
                {
                    prob := previous_probability + arm_prob
                    if rand_float < prob
                    {
                        chosen_arm_index = i
                        break
                    }
                    previous_probability = prob
                }

                chosen_arm := &arms[chosen_arm_index]
                chosen_arm_reward := rand.float64_normal(mean = chosen_arm.true_value, stddev = 1)

                if steps_since_measurement_start := k - PERFORMANCE_MEASUREMENT_START_STEPS; steps_since_measurement_start > 0
                {                   
                    run_average_reward += (chosen_arm_reward - run_average_reward) / f64(steps_since_measurement_start)
                }

                rewards_average += (chosen_arm_reward - rewards_average) / f64(k)

                for &arm, i in arms
                {
                    arm_prob := arm_probabilities[i]
                    if chosen_arm_index == i
                    {
                        arm.preference_value += alpha * (chosen_arm_reward - rewards_average) * ( 1 - arm_prob)
                    } else
                    {
                        arm.preference_value -= alpha * (chosen_arm_reward - rewards_average) * arm_prob
                    }
                }
            }

            free_all(context.temp_allocator)
            gradient_reward_average[i] += (run_average_reward - gradient_reward_average[i]) / f64(j)
        }
    }

    Q_VALUES :: []f64 { 1/4.0, 1/2.0, 1, 2, 4}
    optimistic_greedy_reward_average := make([]f64, len(Q_VALUES))
    for q_value, i in Q_VALUES
    {
        ALPHA :: 0.1
        for j in 1 ..= RUNS
        {
            run_average_reward: f64

            // select starting action values
            true_value := rand.float64_normal(mean = 0, stddev = 1)
            arms: [ARMS]Arm
            for &arm in arms
            {
                arm.true_value = true_value
                arm.estimated_value = q_value
            }
            rewards_average: f64

            for k in 0 ..< STEPS
            {
                // update arm every step to simulate a nonstationary problem
                for &arm in arms
                {
                    arm.true_value += rand.float64_normal(mean = 0, stddev = 0.01)
                }

                greedy_options := make([dynamic]int, context.temp_allocator)
                best_value := min(f64)
                for arm, arm_index in arms
                {
                    if arm.estimated_value >= best_value
                    {
                        if arm.estimated_value > best_value
                        {
                            clear(&greedy_options)
                            best_value = arm.estimated_value
                        }
                        append(&greedy_options, arm_index)
                    }
                }
                chosen_arm_index := rand.choice(greedy_options[:])
                chosen_arm := &arms[chosen_arm_index]

                chosen_arm_reward := rand.float64_normal(mean = chosen_arm.true_value, stddev = 1)
                chosen_arm.estimated_value += ALPHA * (chosen_arm_reward - chosen_arm.estimated_value)

                if steps_since_measurement_start := k - PERFORMANCE_MEASUREMENT_START_STEPS; steps_since_measurement_start > 0
                {                   
                    run_average_reward += (chosen_arm_reward - run_average_reward) / f64(steps_since_measurement_start)
                }
            }

            free_all(context.temp_allocator)
            optimistic_greedy_reward_average[i] += (run_average_reward - optimistic_greedy_reward_average[i]) / f64(j)
        }
    }

    C_VALUES :: []f64 { 1/16.0, 1/8.0, 1/4.0, 1/2.0, 1, 2, 4}
    ucb_reward_average := make([]f64, len(C_VALUES))
    for c_value, i in C_VALUES
    {
        for j in 1 ..= RUNS
        {
            run_average_reward: f64

            // select starting action values
            true_value := rand.float64_normal(mean = 0, stddev = 1)
            arms: [ARMS]Arm
            for &arm in arms
            {
                arm.true_value = true_value
            }

            for k in 0 ..< STEPS
            {
                // update arm every step to simulate a nonstationary problem
                for &arm in arms
                {
                    arm.true_value += rand.float64_normal(mean = 0, stddev = 0.01)
                }

                maximizing_options := make([dynamic]int, context.temp_allocator)
                never_taken_arms := make([dynamic]int, context.temp_allocator)
                max_ucb := min(f64)
                for arm, i in arms
                {
                    if arm.times_taken == 0
                    {
                        append(&never_taken_arms, i)
                    }

                    if len(never_taken_arms) > 0
                    {
                        continue
                    }

                    ucb_val := arm.estimated_value + (c_value * math.sqrt(math.ln(f64(k)) / f64(arm.times_taken)))
                    if ucb_val >= max_ucb
                    {
                        if ucb_val > max_ucb
                        {
                            max_ucb = ucb_val
                            clear(&maximizing_options)
                        }
                        append(&maximizing_options, i)
                    }
                }

                chosen_arm_index: int
                if len(never_taken_arms) > 0
                {
                    chosen_arm_index = rand.choice(never_taken_arms[:])
                } else
                {
                    chosen_arm_index = rand.choice(maximizing_options[:])
                }
                
                chosen_arm := &arms[chosen_arm_index]
                chosen_arm.times_taken += 1
        
                chosen_arm_reward := rand.float64_normal(mean = chosen_arm.true_value, stddev = 1)

                if steps_since_measurement_start := k - PERFORMANCE_MEASUREMENT_START_STEPS; steps_since_measurement_start > 0
                {                   
                    run_average_reward += (chosen_arm_reward - run_average_reward) / f64(steps_since_measurement_start)
                }

                chosen_arm.estimated_value += 1 / f64(chosen_arm.times_taken) * (chosen_arm_reward - chosen_arm.estimated_value)                
            }

            free_all(context.temp_allocator)
            ucb_reward_average[i] += (run_average_reward - ucb_reward_average[i]) / f64(j)
        }
    }
    
    eps_avg_rewards_x := make([]f64, len(EPSILON_VALUES))
    eps_avg_rewards_y := epsilon_greedy_reward_average
    for epsilon, i in EPSILON_VALUES
    {
        eps_avg_rewards_x[i] = epsilon
    }

    gb_avg_rewards_x := make([]f64, len(ALPHA_VALUES))
    gb_avg_rewards_y := gradient_reward_average
    for alpha, i in ALPHA_VALUES
    {
        gb_avg_rewards_x[i] = alpha
    }

    og_avg_rewards_x := make([]f64, len(Q_VALUES))
    og_avg_rewards_y := optimistic_greedy_reward_average
    for q_value, i in Q_VALUES
    {
        og_avg_rewards_x[i] = q_value
    }
        
    ucb_avg_rewards_x := make([]f64, len(C_VALUES))
    ucb_avg_rewards_y := ucb_reward_average
    for c_value, i in C_VALUES
    {
        ucb_avg_rewards_x[i] = c_value
    }
    

    average_rewards_plots := []Plot {
        Plot {
          label = "ε-greedy",
          x_data = eps_avg_rewards_x,
          y_data = eps_avg_rewards_y,
        },
        Plot {
          label = "Gradient Bandit",
          x_data = gb_avg_rewards_x,
          y_data = gb_avg_rewards_y,
        },
        Plot {
            label = "Optimistic Greedy",
            x_data = og_avg_rewards_x,
            y_data = og_avg_rewards_y,
        },
        Plot {
            label = "UCB",
            x_data = ucb_avg_rewards_x,
            y_data = ucb_avg_rewards_y,
        }
    }

    average_rewards_figure := Figure {
        plots = average_rewards_plots[:],
        x_label = "ε, α, Q0, c",
        y_label = "Average Reward over last 100,000 steps",
        title = "Bandit Algorithms Compared with Different Configurations"
    }

    figures:= []Figure { average_rewards_figure }
    output := Output { figures }
    marshalled_figure_data, marshalled_figure_data_ok := json.marshal(output)
    assert(marshalled_figure_data_ok == nil)

    figures_json_file, figures_json_file_err := os2.create("build/figures.json")
    assert(figures_json_file_err == nil)
    os2.write(figures_json_file, marshalled_figure_data)    
}
