import csv


def parse_float_list(value):
	return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value):
	return [int(x.strip()) for x in value.split(",") if x.strip()]


def save_search_results(rows, output_path):
	if not rows:
		return

	fieldnames = [
		"search_mode",
		"trial",
		"lr",
		"hidden_dim",
		"weight_decay",
		"epoch",
		"effective_lr",
		"train_loss",
		"train_acc",
		"val_loss",
		"val_acc",
	]

	with open(output_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print(f"Saved search results to: {output_path}")