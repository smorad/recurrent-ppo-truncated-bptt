import numpy as np
import os
import pickle
import torch
import time
from torch import optim
from buffer import Buffer
from model import ActorCriticModel, PhiNet
from worker import Worker
from utils import create_env
from utils import polynomial_decay
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class PPOTrainer:
    def __init__(
        self,
        config: dict,
        run_id: str = "run",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initializes all needed training components.

        Args:
            config {dict} -- Configuration and hyperparameters of the environment,
            trainer and model.

            run_id {str, optional} -- A tag used to save Tensorboard Summaries and
            the trained model. Defaults to "run".

            device {torch.device, optional} -- Determines the training device.
            Defaults to cpu.
        """
        # Set variables
        self.config = config
        self.recurrence = config["recurrence"]
        self.device = device
        self.run_id = run_id
        self.lr_schedule = config["learning_rate_schedule"]
        self.beta_schedule = config["beta_schedule"]
        self.cr_schedule = config["clip_range_schedule"]

        # Setup Tensorboard Summary Writer
        if not os.path.exists("./summaries"):
            os.makedirs("./summaries")
        timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        self.writer = SummaryWriter("./summaries/" + run_id + timestamp)

        # Init dummy environment and retrieve action and observation spaces
        print("Step 1: Init dummy environment")
        dummy_env = create_env(self.config["env"])
        observation_space = dummy_env.observation_space
        action_space_shape = (dummy_env.action_space.n,)
        dummy_env.close()

        # Init buffer
        print("Step 2: Init buffer")
        self.buffer = Buffer(self.config, observation_space, self.device)

        # Init model
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(
            self.config, observation_space, action_space_shape
        ).to(self.device)
        self.phi = PhiNet(
            self.model.in_features_next_layer, self.recurrence["hidden_state_size"]
        )
        self.model.train()
        self.phi.train()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr_schedule["initial"]
        )
        self.phi_optimizer = optim.AdamW(self.phi.parameters(), lr=0.001)

        # Init workers
        print("Step 4: Init environment workers")
        self.workers = [
            Worker(self.config["env"]) for w in range(self.config["n_workers"])
        ]

        # Setup observation placeholder
        self.obs = np.zeros(
            (self.config["n_workers"],) + observation_space.shape, dtype=np.float32
        )

        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        self.recurrent_cell = self.model.init_recurrent_cell_states(
            self.config["n_workers"], self.device
        )

        # Reset workers (i.e. environments)
        print("Step 5: Reset workers")
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholder location
        for w, worker in enumerate(self.workers):
            self.obs[w] = worker.child.recv()

    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model."""
        print("Step 6: Starting training")
        # Store episode results for monitoring statistics
        episode_infos = deque(maxlen=100)

        for update in range(self.config["updates"]):
            # Decay hyperparameters polynomially based on the provided config
            learning_rate = polynomial_decay(
                self.lr_schedule["initial"],
                self.lr_schedule["final"],
                self.lr_schedule["max_decay_steps"],
                self.lr_schedule["power"],
                update,
            )
            beta = polynomial_decay(
                self.beta_schedule["initial"],
                self.beta_schedule["final"],
                self.beta_schedule["max_decay_steps"],
                self.beta_schedule["power"],
                update,
            )
            clip_range = polynomial_decay(
                self.cr_schedule["initial"],
                self.cr_schedule["final"],
                self.cr_schedule["max_decay_steps"],
                self.cr_schedule["power"],
                update,
            )

            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            self.buffer.prepare_batch_dict()

            # Train epochs
            training_stats = self._train_epochs(learning_rate, clip_range, beta)
            training_stats = np.mean(training_stats, axis=0)

            # Store recent episode infos
            episode_infos.extend(sampled_episode_info)
            episode_result = self._process_episode_info(episode_infos)

            # Print training statistics
            if "success_percent" in episode_result:
                result = (
                    "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success ="
                    " {:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f}"
                    " phi_loss={:3f} "
                    " value={:.3f} advantage={:.3f}".format(
                        update,
                        episode_result["reward_mean"],
                        episode_result["reward_std"],
                        episode_result["length_mean"],
                        episode_result["length_std"],
                        episode_result["success_percent"],
                        training_stats[0],
                        training_stats[1],
                        training_stats[3],
                        training_stats[2],
                        training_stats[4],
                        torch.mean(self.buffer.values),
                        torch.mean(self.buffer.advantages),
                    )
                )
            else:
                result = (
                    "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f}"
                    " pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} "
                    " phi_loss={:3f} "
                    "value={:.3f}  advantage={:.3f}".format(
                        update,
                        episode_result["reward_mean"],
                        episode_result["reward_std"],
                        episode_result["length_mean"],
                        episode_result["length_std"],
                        training_stats[0],
                        training_stats[1],
                        training_stats[3],
                        training_stats[2],
                        training_stats[4],
                        torch.mean(self.buffer.values),
                        torch.mean(self.buffer.advantages),
                    )
                )
            print(result)

            # Write training statistics to tensorboard
            self._write_training_summary(update, training_stats, episode_result)

        # Save the trained model at the end of the training
        self._save_model()

    def _sample_training_data(self) -> list:
        """Runs all n workers for n steps to sample training data.

        Returns:
            {list} -- list of results of completed episodes.
        """
        episode_infos = []
        # Sample actions from the model and collect experiences for training
        for t in range(self.config["worker_steps"]):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # Save the initial observations and recurrentl cell states
                self.buffer.obs[:, t] = torch.tensor(self.obs)
                # Safely modify the dict we iterate over
                for i, hs_name in enumerate(list(self.buffer.hidden_states.keys())):
                    self.buffer.hidden_states[hs_name][:, t] = self.recurrent_cell[
                        i
                    ].squeeze(0)

                # Forward the model to retrieve the policy, the states' value
                # and the recurrent cell states
                policy, value, self.recurrent_cell = self.model(
                    torch.tensor(self.obs), self.recurrent_cell, self.device
                )

                self.buffer.values[:, t] = value

                # Sample actions
                action = policy.sample()
                log_prob = policy.log_prob(action)
                self.buffer.actions[:, t] = action
                self.buffer.log_probs[:, t] = log_prob

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", self.buffer.actions[w, t].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                (
                    obs,
                    self.buffer.rewards[w, t],
                    self.buffer.dones[w, t],
                    info,
                ) = worker.child.recv()
                if info:
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    # Reset agent (potential interface for providing reset parameters)
                    worker.child.send(("reset", None))
                    # Get data from reset
                    obs = worker.child.recv()
                    # Reset recurrent cell states
                    if self.recurrence["reset_hidden_state"]:
                        hidden_states = self.model.init_recurrent_cell_states(
                            1, self.device
                        )
                        for i in range(len(hidden_states)):
                            self.recurrent_cell[i][:, w] = hidden_states[i]

                # Store latest observations
                self.obs[w] = obs

        # Calculate advantages
        _, last_value, _ = self.model(
            torch.tensor(self.obs), self.recurrent_cell, self.device
        )
        self.buffer.calc_advantages(
            last_value, self.config["gamma"], self.config["lamda"]
        )

        return episode_infos

    def _train_epochs(
        self, learning_rate: float, clip_range: float, beta: float
    ) -> list:
        """Trains several PPO epochs over one batch of data while
        dividing the batch into mini batches.

        Args:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient

        Returns:
            {list} -- Training statistics of one training epoch"""
        train_info = []
        for _ in range(self.config["epochs"]):
            # Retrieve the to be trained mini batches via a generator
            mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            for mini_batch in mini_batch_generator:
                train_info.append(
                    self._train_mini_batch(mini_batch, learning_rate, clip_range, beta)
                )
        return train_info

    def _compute_phi_loss(self, samples, num_dropouts=4):
        ####
        # Overview:
        #
        # 1. Select a dropout idx i for each time sequence, where
        # 0 <= i_sequence <= seq_lens[sequence]
        #
        # 2. Run obs through encoder and LSTM to compute
        # LSTM hidden state at time i-1
        #
        # 3. Compute values for i..n with hidden state i-1
        # and values for i..n with hidden state i. We call these
        # the "divergent" values, because the value functions diverge
        # at i
        #
        # 4. Compute loss between divergent and normal values
        ####
        inputs = []
        targets = []

        ## Part 1.
        #
        # Chop into sequences
        start_shape = [
            samples["obs"].shape[0] // self.recurrence["sequence_length"],
            self.recurrence["sequence_length"],
        ]
        # Remove all sequences less than 3 timesteps long
        # 2 timesteps needed to have a divergence
        # at o_0 we see m=null and x=o_0
        # at o_1 we see m=null and x=o_1 (no more o_0 visible)
        batch_mask = samples["loss_mask"].reshape(start_shape).sum(dim=-1) > 1
        seq_shape = [
            samples["obs"].shape[0] // self.recurrence["sequence_length"],
            self.recurrence["sequence_length"],
        ]
        # Make sure we mask out padding
        seq_mask = samples["loss_mask"].reshape(start_shape).bool()[batch_mask]
        # Reshape into sequences [B, T, x]
        obs_seq = samples["obs"].reshape(start_shape + [samples["obs"].shape[-1]])[batch_mask]
        # The non-padded length of each sequence
        seq_lens = seq_mask.sum(dim=1).int()

        for d in range(num_dropouts):
            with torch.no_grad():
                # Compute the index to dropout (randomly)
                # -1 so we don't sample the final step in a segment
                dropout_idx = torch.randint(
                    self.recurrence["sequence_length"] - 1, seq_lens.shape
                )
                # Some dropout idx may refer to segments shorter than
                # sequence_length, fix overflows using moduluo
                # With n elems, we get n-1 from indexing and
                # n-2 because we don't want the last step in an sequence
                # n-3 because we don't see the effect until the step
                # after, because o_t is seen at o_t
                dropout_idx = dropout_idx % (seq_lens - 1)

                ## Part 2
                #
                # z will be used to compute values later
                z = self.model.encoder(obs_seq)
                # batch size
                B = z.shape[0]
                # Compute the convergent and divergent masks
                # Convergent is from 0 to seq_len
                # Divergent is from 0 to seq_len, not including i
                # Diff mask is i..seq_len
                select_idx = torch.arange(
                    0, self.recurrence["sequence_length"]
                ).expand(B, self.recurrence["sequence_length"])
                convergent_seq_lens = dropout_idx
                convergent_mask = seq_mask
                divergent_mask = (
                    select_idx != dropout_idx.unsqueeze(1)
                ) * seq_mask
                # Full sequence is all three masks, while the dropout
                # sequence removes the dropout mask
                convergent_output, _ = self.model.recurrent_layer(
                    z,
                    state=self.model.init_recurrent_cell_states(B, z.device),
                    mask=convergent_mask
                )
                divergent_output, _ = self.model.recurrent_layer(
                    z,
                    state=self.model.init_recurrent_cell_states(B, z.device),
                    mask=divergent_mask
                )
                # Value estimates
                _, convergent_v = self.model.policy(convergent_output)
                _, divergent_v = self.model.policy(divergent_output)
                # Compute error/residual
                convergent_v = convergent_v.reshape(convergent_mask.shape)
                divergent_v = divergent_v.reshape(convergent_mask.shape)
                # Sanity check
                residual = convergent_v - divergent_v
                # Note that we do memory(m, o_t), so we
                # need to compare starting one step in the future
                # (when o_t is not present in memory and also not present
                # in obs). For this reason, ignore the first masked value.
                residual_mask = (select_idx > dropout_idx.unsqueeze(1)) * seq_mask
                """
                assert torch.all(
                    convergent_v[residual_mask] 
                    != divergent_v[residual_mask]
                )
                """
                # Zero invalid (masked-out) entries
                # as well as entries they have in common.
                masked_residual = residual.masked_scatter(
                    ~residual_mask, torch.zeros(residual_mask.shape)
                )
                # Phi should predict the absolute mean error
                target = (
                    masked_residual.abs().sum(dim=-1) 
                    / masked_residual.count_nonzero(dim=-1)
                )
                # Now add train data (X, Y, and mask)
                inputs.append(z[torch.arange(B), dropout_idx])
                targets.append(target)

        # Compute loss and train
        x = torch.cat(inputs)
        y = torch.cat(targets).unsqueeze(1)
        y_hat = self.phi(x)

        error = y - y_hat
        loss = (error **  2).mean()
        loss_std = error.std().detach()
        pred_mean = y_hat.mean().detach()
        pred_std = y_hat.std().detach()

        return loss, loss_std, pred_mean, pred_std

    def _train_mini_batch(
        self, samples: dict, learning_rate: float, clip_range: float, beta: float
    ) -> list:
        """Uses one mini batch to optimize the model.

        Args:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient

        Returns:
            {list} -- list of trainig statistics (e.g. loss)
        """
        # Retrieve sampled recurrent cell states to feed the model
        if self.recurrence["layer_type"] == "gru":
            recurrent_cell = samples["hxs"].unsqueeze(0)
        elif self.recurrence["layer_type"] == "lstm":
            recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))

        # Forward model
        policy, value, _ = self.model(
            samples["obs"],
            recurrent_cell,
            self.device,
            self.recurrence["sequence_length"],
        )

        # Compute policy surrogates to establish the policy loss
        normalized_advantage = (
            samples["advantages"] - samples["advantages"].mean()
        ) / (samples["advantages"].std() + 1e-8)
        log_probs = policy.log_prob(samples["actions"])
        ratio = torch.exp(log_probs - samples["log_probs"])
        surr1 = ratio * normalized_advantage
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            * normalized_advantage
        )
        policy_loss = torch.min(surr1, surr2)
        policy_loss = PPOTrainer._masked_mean(policy_loss, samples["loss_mask"])

        # Value  function loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(
            min=-clip_range, max=clip_range
        )
        vf_loss = torch.max(
            (value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2
        )
        vf_loss = PPOTrainer._masked_mean(vf_loss, samples["loss_mask"])

        # Entropy Bonus
        entropy_bonus = PPOTrainer._masked_mean(policy.entropy(), samples["loss_mask"])

        """
        phi_loss, phi_loss_std, pred_mean, pred_std = self._compute_phi_loss(
            samples
        )
        """

        # Complete loss
        loss = -(
            policy_loss
            - self.config["value_loss_coefficient"] * vf_loss
            + beta * entropy_bonus
        )

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        """
        self.phi_optimizer.zero_grad()
        phi_loss.backward()
        self.optimizer.step()
        """

        return [
            policy_loss.cpu().data.numpy(),
            vf_loss.cpu().data.numpy(),
            loss.cpu().data.numpy(),
            entropy_bonus.cpu().data.numpy(),
            0,0,0,0
            #phi_loss.cpu().data.numpy(),
            #phi_loss_std.cpu().data.numpy(),
            #pred_mean.cpu().data.numpy(),
            #pred_std.cpu().data.numpy(),
        ]

    def _write_training_summary(self, update, training_stats, episode_result) -> None:
        """Writes to an event file based on the run-id argument.

        Args:
            update {int} -- Current PPO Update
            training_stats {list} -- Statistics of the training algorithm
            episode_result {dict} -- Statistics of completed episodes
        """
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar(
                        "episode/" + key, episode_result[key], update
                    )
        self.writer.add_scalar("losses/loss", training_stats[2], update)
        self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
        self.writer.add_scalar("losses/value_loss", training_stats[1], update)
        self.writer.add_scalar("losses/entropy", training_stats[3], update)
        self.writer.add_scalar("losses/phi_loss", training_stats[4], update)
        self.writer.add_scalar("losses/phi_loss_std", training_stats[5], update)
        self.writer.add_scalar("losses/phi_pred_mean", training_stats[6], update)
        self.writer.add_scalar("losses/phi_pred_std", training_stats[7], update)
        self.writer.add_scalar(
            "training/sequence_length", self.buffer.true_sequence_length, update
        )
        self.writer.add_scalar(
            "training/value_mean", torch.mean(self.buffer.values), update
        )
        self.writer.add_scalar(
            "training/advantage_mean", torch.mean(self.buffer.advantages), update
        )

    @staticmethod
    def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean of the tensor but ignores the values specified by the mask.
        This is used for masking out the padding of the loss functions.

        Args:
            tensor {Tensor} -- The to be masked tensor
            mask {Tensor} -- The mask that is used to mask out
            padded values of a loss function

        Returns:
            {Tensor} -- Returns the mean of the masked tensor.
        """
        return (tensor.T * mask).sum() / torch.clamp(
            (torch.ones_like(tensor.T) * mask).float().sum(), min=1.0
        )

    @staticmethod
    def _process_episode_info(episode_info: list) -> dict:
        """Extracts the mean and std of completed episode statistics like length and total reward.

        Args:
            episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

        Returns:
            {dict} -- Processed episode results (computes the mean and std for most available keys)
        """
        result = {}
        if len(episode_info) > 0:
            for key in episode_info[0].keys():
                if key == "success":
                    # This concerns the PocMemoryEnv only
                    episode_result = [info[key] for info in episode_info]
                    result[key + "_percent"] = np.sum(episode_result) / len(
                        episode_result
                    )
                result[key + "_mean"] = np.mean([info[key] for info in episode_info])
                result[key + "_std"] = np.std([info[key] for info in episode_info])
        return result

    def _save_model(self) -> None:
        """Saves the model and the used training config to the models directory. The filename is based on the run id."""
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.model.cpu()
        pickle.dump(
            (self.model.state_dict(), self.config),
            open("./models/" + self.run_id + ".nn", "wb"),
        )
        print("Model saved to " + "./models/" + self.run_id + ".nn")

    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        try:
            self.dummy_env.close()
        except:
            pass

        try:
            self.writer.close()
        except:
            pass

        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        time.sleep(1.0)
        exit(0)
