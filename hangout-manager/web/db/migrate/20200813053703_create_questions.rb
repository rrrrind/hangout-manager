class CreateQuestions < ActiveRecord::Migration[6.0]
  def change
    create_table :questions do |t|
      t.integer :q_one
      t.integer :q_two
      t.integer :q_three

      t.timestamps
    end
  end
end
