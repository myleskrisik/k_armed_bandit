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
    STEPS :: 10_000
    RUNS :: 2000

    EPSILON :: 0.1
    ALPHA :: 0.1

    sample_average_reward := make([]f64, STEPS)
    sample_average_optimal_actions := make([]u32, STEPS)

    for i in 1 ..= RUNS
    {
        // select starting action values
        true_value := rand.float64_normal(mean = 0, stddev = 1)
        arms: [ARMS]Arm
        for &arm in arms
        {
            arm.true_value = true_value
        }

        for j in 0 ..< STEPS
        {
            // update arm every step to simulate a nonstationary problem
            for &arm in arms
            {
                arm.true_value += rand.float64_normal(mean = 0, stddev = 0.01)
            }

            should_explore := rand.float64() < EPSILON

            chosen_arm_index: int
            if should_explore
            {
                chosen_arm_index = rand.int_max(ARMS)
            } else
            {
                greedy_options := make([dynamic]int, context.temp_allocator)
                best_value := min(f64)
                for arm, i in arms
                {
                    if arm.estimated_value >= best_value
                    {
                        if arm.estimated_value > best_value
                        {
                            clear(&greedy_options)
                            best_value = arm.estimated_value
                        }
                        append(&greedy_options, i)
                    }
                }
                chosen_arm_index = rand.choice(greedy_options[:])
            }
            chosen_arm := &arms[chosen_arm_index]

            best_arm_index: int
            best_arm_true_value := min(f64)
            for arm, i in arms
            {
                if arm.true_value > best_arm_true_value
                {
                    best_arm_index = i
                    best_arm_true_value = arm.true_value
                }
            }

            if best_arm := arms[best_arm_index]; best_arm.true_value == chosen_arm.true_value
            {
                sample_average_optimal_actions[j] += 1
            }
            
            chosen_arm.times_taken += 1

            chosen_arm_reward := rand.float64_normal(mean = chosen_arm.true_value, stddev = 1)

            sample_average_reward[j] = sample_average_reward[j] + (chosen_arm_reward - sample_average_reward[j]) / f64(i)

            chosen_arm.estimated_value = chosen_arm.estimated_value + 1 / f64(chosen_arm.times_taken) * (chosen_arm_reward - chosen_arm.estimated_value)
        }
    }

    constant_step_size_reward := make([]f64, STEPS)
    constant_step_size_optimal_actions := make([]u32, STEPS)

    for i in 1 ..= RUNS
    {
        // select starting action values
        true_value := rand.float64_normal(mean = 0, stddev = 1)
        arms: [ARMS]Arm
        for &arm in arms
        {
            arm.true_value = true_value
        }

        for j in 0 ..< STEPS
        {
            // update arm every step to simulate a nonstationary problem
            for &arm in arms
            {
                arm.true_value += rand.float64_normal(mean = 0, stddev = 0.01)
            }
            
            should_explore := rand.float64() < EPSILON

            chosen_arm_index: int
            if should_explore
            {
                chosen_arm_index = rand.int_max(ARMS)
            } else
            {
                greedy_options := make([dynamic]int, context.temp_allocator)
                best_value := min(f64)
                for arm, i in arms
                {
                    if arm.estimated_value >= best_value
                    {
                        if arm.estimated_value > best_value
                        {
                            clear(&greedy_options)
                            best_value = arm.estimated_value
                        }
                        append(&greedy_options, i)
                    }
                }
                chosen_arm_index = rand.choice(greedy_options[:])
            }
            chosen_arm := &arms[chosen_arm_index]

            best_arm_index: int
            best_arm_true_value := min(f64)
            for arm, i in arms
            {
                if arm.true_value > best_arm_true_value
                {
                    best_arm_index = i
                    best_arm_true_value = arm.true_value
                }
            }

            if best_arm := arms[best_arm_index]; best_arm.true_value == chosen_arm.true_value
            {
                constant_step_size_optimal_actions[j] += 1
            }
            
            chosen_arm.times_taken += 1

            chosen_arm_reward := rand.float64_normal(mean = chosen_arm.true_value, stddev = 1)

            constant_step_size_reward[j] = constant_step_size_reward[j] + (chosen_arm_reward - constant_step_size_reward[j]) / f64(i)

            chosen_arm.estimated_value = chosen_arm.estimated_value + ALPHA * (chosen_arm_reward - chosen_arm.estimated_value)
        }
    }
    
    rewards_x_avg := make([]f64, STEPS)
    rewards_y_avg := sample_average_reward
    for i in 0 ..< STEPS
    {
        rewards_x_avg[i] = f64(i)
    }

    rewards_x_constant := make([]f64, STEPS)
    rewards_y_constant := constant_step_size_reward
    for i in 0 ..< STEPS
    {
        rewards_x_constant[i] = f64(i)
    }

    average_rewards_plots := []Plot {
        Plot {
          label = "Sample-Average",
          x_data = rewards_x_avg,
          y_data = rewards_y_avg,
        },
        Plot {
          label = "Constant Step-Size",
          x_data = rewards_x_constant,
          y_data = rewards_y_constant,
        }
    }

    average_rewards_figure := Figure {
        plots = average_rewards_plots[:],
        x_label = "Steps",
        y_label = "Average Reward",
        title = "Average Reward Constant Step-Size vs Sample-Average"
    }

    optimal_actions_x_avg := make([]f64, STEPS)
    optimal_actions_y_avg := make([]f64, STEPS)
    for i in 0 ..< STEPS
    {
        optimal_actions_x_avg[i] = f64(i)
        optimal_actions_y_avg[i] = f64(sample_average_optimal_actions[i]) / RUNS * 100.0
    }

    optimal_actions_x_constant := make([]f64, STEPS)
    optimal_actions_y_constant := make([]f64, STEPS)
    for i in 0 ..< STEPS
    {
        optimal_actions_x_constant[i] = f64(i)
        optimal_actions_y_constant[i] = f64(constant_step_size_optimal_actions[i]) / RUNS * 100.0
    }

    optimal_actions_plots := []Plot {
        Plot {
          label = "Sample-Average",
          x_data = optimal_actions_x_avg,
          y_data = optimal_actions_y_avg,
        },
        Plot {
          label = "Constant Step-Size",
          x_data = optimal_actions_x_constant,
          y_data = optimal_actions_y_constant,
        }
    }

    optimal_actions_figure := Figure {
        plots = optimal_actions_plots[:],
        x_label = "Steps",
        y_label = "% Optimal Actions",
        title = "% of Time Optimal Action Taken Constant Step-Size vs Sample-Average"
    }

    figures:= []Figure { average_rewards_figure, optimal_actions_figure }
    output := Output { figures }
    marshalled_figure_data, marshalled_figure_data_ok := json.marshal(output)
    assert(marshalled_figure_data_ok == nil)

    figures_json_file, figures_json_file_err := os2.create("build/figures.json")
    assert(figures_json_file_err == nil)
    os2.write(figures_json_file, marshalled_figure_data)    
}
