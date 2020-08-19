class CreateUserDetails < ActiveRecord::Migration[6.0]
  def change
    create_table :user_details do |t|
      t.integer :d_one
      t.integer :d_two
      t.integer :d_three
      t.integer :d_four
      t.integer :d_five
      t.integer :d_six
      t.integer :d_seven
      t.integer :d_eight
      t.references :user

      t.timestamps
    end
  end
end
