class CreateUserPersonals < ActiveRecord::Migration[6.0]
  def change
    create_table :user_personals do |t|
      t.integer :p_one
      t.integer :p_two
      t.integer :p_three
      t.integer :p_four
      t.integer :p_five
      t.integer :p_six
      t.references :user

      t.timestamps
    end
  end
end
